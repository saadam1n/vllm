import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

@triton.jit
def hsa_reduce_block(
    key_ptr,
    key_stride_0,
    q_base,
    q_end,
    rblock_idx, 
    FULL_KEY_DIM : tl.constexpr, # number of KV groups * hidden dim 
    R_BLOCK_SIZE : tl.constexpr # reduction block size (used in prefill)
):
    off_tokens = tl.arange(0, R_BLOCK_SIZE) + R_BLOCK_SIZE * rblock_idx
    off_key_dim = tl.arange(0, FULL_KEY_DIM)

    # (BLOCK_SIZE, 1)
    off_tokens = off_tokens[:, None]
    # (1, FULL_KEY_DIM)
    off_key_dim = off_key_dim[None, :]

    ld_mask = (q_base + off_tokens < q_end)

    keys = tl.load(key_ptr + key_stride_0 * (q_base + off_tokens) + off_key_dim, mask=ld_mask, other=float("-inf"))

    rblock_keymax = tl.max(keys, axis=0)

    keys = tl.where(ld_mask, keys, float("inf"))
    rblock_keymin = tl.min(keys, axis=0)

    return rblock_keymax, rblock_keymin

# warmup example
@triton.jit
def hsa_select(
    query_ptr, # [number of queries, number of KV groups, number of querys per group, hidden dim]
    key_ptr, # [number of queries, number of KV groups, hidden dim]
    key_cache_ptr, # [number of blocks, block size, number of KV groups, hidden dim]
    value_cache_ptr, # [number of blocks, block size, number of KV groups, hidden dim]
    seq_lens_ptr, # [batch,]
    block_table_ptr, # [batch, max blocks]
    scratch_table_ptr, # [batch, max blocks]
    query_start_loc_ptr, # [batch + 1,]
    slot_mapping_ptr, # [number of queries,]
    query_stride_0 : int, # stride of query_ptr dim 0
    key_stride_0 : int, # stride of key_ptr dim 0 (needed in some cases)
    kv_stride_1 : int, # corresponds to stride of number of KV groups * hidden dim * sizeof(dtype)
    block_table_stride_0 : int,
    kv_top : int, # token with highest index in KV cache
    MAX_BLOCK_BUDGET : tl.constexpr,
    NUM_KV_GROUPS : tl.constexpr, # number of KV groups
    NUM_Q_PER_GROUP : tl.constexpr, # number of queries per group
    QK_HIDDEN_DIM : tl.constexpr, # hidden dim of queries and keys
    FULL_KEY_DIM : tl.constexpr,
    KV_BLOCK_SIZE : tl.constexpr, # size of page in vLLM
    R_BLOCK_SIZE : tl.constexpr, # reduction block size (used in prefill)
    QKH_BLOCK_SIZE : tl.constexpr, # block size used to eval QK hat
):
    # short note on kv top:
    # we want to store the min and max for each block for hybrid sparse attention
    # we can just repurpose the upper memory of the KV cache 
    # however, KV cache fmt is [blocks, block size, kv heads, h dim]
    # to store min *or* max for an entire block we just need [1, kv heads, h dim]
    # so we flatten the kv cache for its first two dimensions and map kv top - physical block idx
    # vllm does not use phys block 0 so we do not have oob errors in that edge case


    # launch dims:
    # [batch, blocks]
    batch_idx = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)

    # get start and end positions as tensor of size (2,)
    q_start = tl.load(query_start_loc_ptr + batch_idx)
    q_end = tl.load(query_start_loc_ptr + batch_idx + 1)

    q_len = q_end - q_start

    if q_len > 1:
        # PREFILL BRANCH
        # notably, we assume there is no case of spec decode
        # that is, each block begins at the start of the block (no updates, only clears)
        
        # each prefill handles one block
        # this may use a massive amount of smem
        # hopefully the compiler can figure it out
        
        q_base = q_start + KV_BLOCK_SIZE * block_idx

        # iterate through remaining blocks
        while q_base < q_end:

            # iterate over page chunk size
            num_tokens = min(q_base + KV_BLOCK_SIZE, q_end) - q_base
            num_rblocks = (num_tokens - 1) // R_BLOCK_SIZE + 1 # cdiv

            running_keymax, running_keymin = hsa_reduce_block(
                key_ptr=key_ptr,
                key_stride_0=key_stride_0,
                q_base=q_base,
                q_end=q_end,
                rblock_idx=0,
                FULL_KEY_DIM=FULL_KEY_DIM,
                R_BLOCK_SIZE=R_BLOCK_SIZE
            )

            for rblock_idx in range(1, num_rblocks):
                rblock_keymax, rblock_keymin = hsa_reduce_block(
                    key_ptr=key_ptr,
                    key_stride_0=key_stride_0,
                    q_base=q_base,
                    q_end=q_end,
                    rblock_idx=rblock_idx,
                    FULL_KEY_DIM=FULL_KEY_DIM,
                    R_BLOCK_SIZE=R_BLOCK_SIZE
                )

                running_keymax = tl.maximum(running_keymax, rblock_keymax)
                running_keymin = tl.minimum(running_keymin, rblock_keymin)

            slot_idx = tl.load(slot_mapping_ptr + q_base)
            phys_block_idx = slot_idx // KV_BLOCK_SIZE
            # assume sub_block_idx == 0 (see decode for more details)
            cache_slot_idx = kv_top - phys_block_idx 

            cache_max_ptr = key_cache_ptr   + kv_stride_1 * cache_slot_idx
            cache_min_ptr = value_cache_ptr + kv_stride_1 * cache_slot_idx

            tl.store(cache_max_ptr + tl.arange(0, FULL_KEY_DIM), running_keymax)
            tl.store(cache_min_ptr + tl.arange(0, FULL_KEY_DIM), running_keymin)

            q_base += KV_BLOCK_SIZE * tl.num_programs(axis=1) 

        # end of prefill branch, we never do HSA here
        # we assume that the scratch table is filled with the correct mappings from the block table already
        # we only modify the scratch table on decode requests
    else:
        # DECODE BRANCH

        # for the rest of the HSA kernel, we use only one block
        if block_idx != 0:
            return

        # decode
        key =  tl.load(key_ptr + key_stride_0 * q_start + tl.arange(0, FULL_KEY_DIM))

        slot_idx = tl.load(slot_mapping_ptr + q_start)

        # replace with bitshift and mask ops when compiling (hopefully)
        # I also need to update my naming conventions
        last_block_idx = slot_idx // KV_BLOCK_SIZE
        sub_block_idx = slot_idx % KV_BLOCK_SIZE

        # note that block_idx cannot equal zero
        cache_slot_idx = kv_top - last_block_idx 

        cache_max_ptr = key_cache_ptr   + kv_stride_1 * cache_slot_idx
        cache_min_ptr = value_cache_ptr + kv_stride_1 * cache_slot_idx

        if sub_block_idx == 0:

            # clear stored value with current key value
            tl.store(cache_max_ptr + tl.arange(0, FULL_KEY_DIM), key)
            tl.store(cache_min_ptr + tl.arange(0, FULL_KEY_DIM), key)
        else:
            # update stored value
            cached_addr = cache_max_ptr + tl.arange(0, FULL_KEY_DIM)
            running_bounds = tl.load(cached_addr)
            running_bounds = tl.maximum(running_bounds, key)
            tl.store(cached_addr, running_bounds)

            cached_addr = cache_min_ptr + tl.arange(0, FULL_KEY_DIM)
            running_bounds = tl.load(cached_addr)
            running_bounds = tl.minimum(running_bounds, key)
            tl.store(cached_addr, running_bounds)

        # begin the actual HSA kernel
        # we first need to load the query

        # TODO: switch from brute force method to a tiled load
        # might not be necessary if triton is smart enough and does this all in registers
        slot_offset = q_start * query_stride_0
        kv_group_offset = tl.arange(0, NUM_KV_GROUPS) * NUM_Q_PER_GROUP * QK_HIDDEN_DIM
        query_offset = tl.arange(0, NUM_Q_PER_GROUP) * QK_HIDDEN_DIM
        hidden_dim_offset = tl.arange(0, QK_HIDDEN_DIM)

        kv_group_offset = kv_group_offset[:, None, None]
        query_offset = query_offset[None, :, None]
        hidden_dim_offset = hidden_dim_offset[None, None, :]

        # [KV, Q, h_d]
        query = tl.load(query_ptr + slot_offset + kv_group_offset + query_offset + hidden_dim_offset)

        # divide first for numerical stability, then reduce
        # hopefully no underflow (vLLM does inference in BF16 so hopefully not a problem)
        q_dtype = query.dtype # next two ops promote to FP32
        query = query / (NUM_Q_PER_GROUP * NUM_KV_GROUPS)
        query = tl.sum(query, axis=1, dtype=query.dtype) # [KV, h_d]

        query = query.to(q_dtype)



        # we actually have very little interest in treating KV and h_d as separate dimensions and will instead merge them
        query = tl.reshape(query, (1, NUM_KV_GROUPS * QK_HIDDEN_DIM))

        # [NUM_KV_GROUPS * QK_HIDDEN_DIM]
        q_negative = (query < 0)

        seq_len = tl.load(seq_lens_ptr + batch_idx)

        num_kv_blocks = (seq_len - 1) // KV_BLOCK_SIZE + 1 # cdiv
        num_qkh_blocks = (num_kv_blocks - 1) // QKH_BLOCK_SIZE + 1 # cdiv

        pos_inf = 3e5
        neg_inf = -3e5

        running_topk_idxs = tl.full((MAX_BLOCK_BUDGET,), value=0, dtype=tl.int32)
        running_topk_vals = tl.full((MAX_BLOCK_BUDGET,), value=neg_inf, dtype=tl.float32)

        # this code is absolutely disgusting and I need to have better naming conventions
        k_offsets = tl.arange(0, MAX_BLOCK_BUDGET)
        qkh_offsets = tl.arange(0, QKH_BLOCK_SIZE)
        for i in range(num_qkh_blocks):
            # [block]
            log_block_segm = tl.arange(0, QKH_BLOCK_SIZE) + QKH_BLOCK_SIZE * i

            log_block_mask = (log_block_segm < num_kv_blocks)

            phys_block_ids = tl.load(
                block_table_ptr + block_table_stride_0 * batch_idx + log_block_segm,
                mask=log_block_mask, 
                other=0
            )


            # load corresponding page min/max
            cached_minmax_ids = kv_top - phys_block_ids
            cached_minmax_off = kv_stride_1 * cached_minmax_ids[:, None]

            mm_cache_offset = tl.arange(0, NUM_KV_GROUPS * QK_HIDDEN_DIM)
            mm_cache_offset = mm_cache_offset[None, :]

            cache_ld_offsets = cached_minmax_off + mm_cache_offset

            mm_cache_max_ptr = key_cache_ptr   + cache_ld_offsets
            mm_cache_min_ptr = value_cache_ptr + cache_ld_offsets

            log_block_mask_2d = log_block_mask[:, None]

            # [qkh block size, kv groups * h dim]
            k_hat = tl.where(
                q_negative,
                tl.load(mm_cache_min_ptr, mask=log_block_mask_2d),
                tl.load(mm_cache_max_ptr, mask=log_block_mask_2d)
            )

            # [kv groups * h_d, qkh block size]
            k_hat = tl.trans(k_hat)

            # ([kv groups * h_d] -> [1, kv_groups * h_d]) x [kv groups * h_d, qkh block size] = [1, qkh block size]
            qk_hat = tl.dot(query, k_hat)
            qk_hat = tl.reshape(qk_hat, (QKH_BLOCK_SIZE))
            qk_hat = tl.where(log_block_mask, qk_hat, neg_inf)

            # based on compairson with a reference impl using pytorch ops,
            # qk_hat should be correct

            # force last block to be selected
            qk_hat = tl.where(log_block_segm == num_kv_blocks - 1, pos_inf, qk_hat)



            # update running topk
            temp_topk_vals = qk_hat
            temp_topk_idxs = phys_block_ids

            resv_topk_vals = running_topk_vals
            resv_topk_idxs = running_topk_idxs

            # dirty trick for top-k 
            # bug: for some cases, we may end up repeating a block ID in the output
            for j in range(MAX_BLOCK_BUDGET):
                # fetch from temp
                max_val_temp = tl.max(temp_topk_vals, axis=0)
                max_idx_temp = tl.argmax(temp_topk_vals, axis=0)

                # fetch from resorvoir
                max_val_resv = tl.max(resv_topk_vals, axis=0)
                max_idx_resv = tl.argmax(resv_topk_vals, axis=0)

                max_idx = 0
                max_val = 0.0

                if max_val_temp > max_val_resv:
                    # mask out temp array
                    k_off_mask_temp = (qkh_offsets == max_idx_temp)
                    temp_topk_vals = tl.where(k_off_mask_temp, neg_inf, temp_topk_vals)

                    max_idx = tl.sum(tl.where(k_off_mask_temp, temp_topk_idxs, 0))
                    max_val = max_val_temp

                    # replace with zero sentinel
                    temp_topk_idxs = tl.where(k_off_mask_temp, 0, temp_topk_idxs)
                else:
                    # mask our resv array
                    k_off_mask_resv = (k_offsets == max_idx_resv)
                    resv_topk_vals = tl.where(k_off_mask_resv, neg_inf, resv_topk_vals)

                    max_idx = tl.sum(tl.where(k_off_mask_resv, resv_topk_idxs, 0))
                    max_val = max_val_resv

                    # replace with zero sentinel
                    resv_topk_idxs = tl.where(k_off_mask_resv, 0, resv_topk_idxs)

                running_topk_idxs = tl.where(k_offsets == j, max_idx, running_topk_idxs)
                running_topk_vals = tl.where(k_offsets == j, max_val, running_topk_vals)

        # now we have the top-k array, just need to flush it out to memory
        # but we need to make sure that the last logical block comes last in the scratch table
        # our top-k kernel will make sure it comes first in running_topk_idxs
        # so we just need to "reverse" it will accounting for the edge case

        actual_block_budget = min(MAX_BLOCK_BUDGET, num_kv_blocks)

        flush_k_offset = tl.arange(0, MAX_BLOCK_BUDGET)

        flush_k_offset = tl.where(
            flush_k_offset < actual_block_budget, 
            actual_block_budget - flush_k_offset - 1, 
            flush_k_offset
        )

        tl.store(scratch_table_ptr + block_table_stride_0 * batch_idx + flush_k_offset, running_topk_idxs)


def select_top_pages(
    query : torch.Tensor,
    key : torch.Tensor,
    key_cache : torch.Tensor,
    value_cache : torch.Tensor,
    seq_lens : torch.Tensor,
    block_table : torch.Tensor,
    scratch_table : torch.Tensor,
    query_start_loc : torch.Tensor,
    slot_mapping : torch.Tensor,
    max_block_budget : int
):
    assert query[0].is_contiguous(), "HSA kernels currently have limited support for non-contiguous tensors"
    assert key[0].is_contiguous(), "HSA kernels currently have limited support for non-contiguous tensors"
    assert key_cache.is_contiguous(), "HSA kernels currently have limited support for non-contiguous tensors"
    assert value_cache.is_contiguous(), "HSA kernels currently have limited support for non-contiguous tensors"
    assert seq_lens.is_contiguous(), "HSA kernels currently have limited support for non-contiguous tensors"
    assert block_table.is_contiguous(), "HSA kernels currently have limited support for non-contiguous tensors"
    assert scratch_table.is_contiguous(), "HSA kernels currently have limited support for non-contiguous tensors"
    assert query_start_loc.is_contiguous(), "HSA kernels currently have limited support for non-contiguous tensors"
    assert slot_mapping.is_contiguous(), "HSA kernels currently have limited support for non-contiguous tensors"

    # key_cache: [num_blocks, block_size, num_kv_groups, hidden_dim]
    kv_top        = key_cache.shape[0] * key_cache.shape[1]
    KV_BLOCK_SIZE = key_cache.shape[1]
    NUM_KV_GROUPS = key_cache.shape[2]
    QK_HIDDEN_DIM = key_cache.shape[3]
    FULL_KEY_DIM = NUM_KV_GROUPS * QK_HIDDEN_DIM

    # query: [num_tokens, num_kv_groups * num_q_per_group, hidden_dim]
    NUM_Q_PER_GROUP = query.shape[1] // NUM_KV_GROUPS

    assert KV_BLOCK_SIZE % 4 == 0
    R_BLOCK_SIZE = KV_BLOCK_SIZE // 4

    # QKH_BLOCK_SIZE controls how many KV pages we score per inner loop iteration.
    # 16 is a reasonable default: large enough to amortize loop overhead and
    # keep the tl.dot tile non-trivial, small enough to stay in registers.
    QKH_BLOCK_SIZE = 4

    # prefix sum increases dim by 1, so we need to subtract 1
    num_reqs   = query_start_loc.shape[0] - 1
    num_blocks = 16
    grid = (num_reqs, num_blocks)

    hsa_select[grid](
        query_ptr               = query,
        key_ptr                 = key,
        key_cache_ptr           = key_cache,
        value_cache_ptr         = value_cache,
        seq_lens_ptr            = seq_lens,
        block_table_ptr         = block_table,
        scratch_table_ptr       = scratch_table,
        query_start_loc_ptr     = query_start_loc,
        slot_mapping_ptr        = slot_mapping,
        query_stride_0          = query.stride(0),
        key_stride_0            = key.stride(0),
        kv_stride_1             = key_cache.stride(1),
        block_table_stride_0    = block_table.stride(0),
        kv_top                  = kv_top,
        MAX_BLOCK_BUDGET        = max_block_budget,
        NUM_KV_GROUPS           = NUM_KV_GROUPS,
        NUM_Q_PER_GROUP         = NUM_Q_PER_GROUP,
        QK_HIDDEN_DIM           = QK_HIDDEN_DIM,
        FULL_KEY_DIM            = FULL_KEY_DIM,
        KV_BLOCK_SIZE           = KV_BLOCK_SIZE,
        R_BLOCK_SIZE            = R_BLOCK_SIZE,
        QKH_BLOCK_SIZE          = QKH_BLOCK_SIZE,
    )