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
    TOP_K_1 : tl.constexpr,
    TOP_K_2 : tl.constexpr, # additional buffer space for k_1
    NUM_Q_PER_HEAD : tl.constexpr, # number of queries per group
    HIDDEN_DIM : tl.constexpr, # hidden dim of queries and keys
    VLLM_BLOCK_SIZE : tl.constexpr, # size of page in vLLM
    POOL_BLOCK_SIZE : tl.constexpr, # how many tokens we pool together
    TOPK_BLOCK_SIZE : tl.constexpr, # block size used to eval QK hat
    debug_ptr,
    debug_stride_0 : tl.constexpr,
):
    # short note on kv top:
    # we want to store the min and max for each block for hybrid sparse attention
    # we can just repurpose the upper memory of the KV cache 
    # however, KV cache fmt is [blocks, block size, kv heads, h dim]
    # to store min *or* max for an entire block we just need [1, kv heads, h dim]
    # so we flatten the kv cache for its first two dimensions and map kv top - physical block idx
    # vllm does not use phys block 0 so we do not have oob errors in that edge case


    # launch dims:
    # [batch, blocks, head idx]
    batch_idx = tl.program_id(axis=0)
    head_idx = tl.program_id(axis=1)

    # only used for prefill
    segm_idx = tl.program_id(axis=2)

    # get start and end positions as tensor of size (2,)
    q_start = tl.load(query_start_loc_ptr + batch_idx)
    q_end = tl.load(query_start_loc_ptr + batch_idx + 1)

    q_len = q_end - q_start

    hd_offsets = HIDDEN_DIM * head_idx + tl.arange(0, HIDDEN_DIM)

    # offset all relevant pointers 
    block_table_ptr = block_table_ptr + block_table_stride_0 * batch_idx
    scratch_table_ptr = scratch_table_ptr + block_table_stride_0 * batch_idx
    seq_lens_ptr = seq_lens_ptr + batch_idx

    debug_ptr = debug_ptr + batch_idx * debug_stride_0

    if q_len == 0:
        return # empty batch
    elif q_len > 1:
        # PREFILL BRANCH
        # notably, we assume there is no case of spec decode
        # that is, each block begins at the start of the block (no updates, only clears)
        
        # each prefill handles one block
        # this may use a massive amount of smem
        # hopefully the compiler can figure it out
        
        q_base = q_start + POOL_BLOCK_SIZE * segm_idx


        # iterate through remaining blocks in prefill
        while q_base < q_end:

            # iterate over page chunk size
            tile_end = min(q_base + POOL_BLOCK_SIZE, q_end)

            # load entire pooling block
            token_offsets = tl.arange(0, POOL_BLOCK_SIZE) + q_base
            token_offsets = token_offsets[:, None]

            ld_mask = (token_offsets < tile_end)

            key_segm_ptr = key_ptr + key_stride_0 * token_offsets + hd_offsets[None, :]

            raw_keys = tl.load(key_segm_ptr, mask=ld_mask, other=float("-inf"))

            mmc_block_max = tl.max(raw_keys, axis=0)

            raw_keys = tl.where(ld_mask, raw_keys, float("inf"))
            mmc_block_min = tl.min(raw_keys, axis=0)

            # slot idx contains the raw token (not block) offset into the KV cache for tokens
            slot_idx = tl.load(slot_mapping_ptr + q_base)
            mmc_block_idx = kv_top - slot_idx // POOL_BLOCK_SIZE

            mmc_max_ptr = key_cache_ptr   + kv_stride_1 * mmc_block_idx + hd_offsets
            mmc_min_ptr = value_cache_ptr + kv_stride_1 * mmc_block_idx + hd_offsets

            tl.store(mmc_max_ptr, mmc_block_max)
            tl.store(mmc_min_ptr, mmc_block_min)

            q_base += POOL_BLOCK_SIZE * tl.num_programs(axis=2) 

        # end of prefill branch, we never do HSA here
        # we assume that the scratch table is filled with the correct mappings from the block table already
        # we only modify the scratch table on decode requests
    else:
        # DECODE BRANCH

        # for the rest of the HSA kernel, we use only one block
        if segm_idx != 0:
            return

        # decode
        key =  tl.load(key_ptr + key_stride_0 * q_start + hd_offsets)

        slot_idx = tl.load(slot_mapping_ptr + q_start)

        mmc_block_idx = kv_top - slot_idx // POOL_BLOCK_SIZE
        mmc_token_idx = slot_idx % POOL_BLOCK_SIZE

        mmc_max_ptr = key_cache_ptr   + kv_stride_1 * mmc_block_idx + hd_offsets
        mmc_min_ptr = value_cache_ptr + kv_stride_1 * mmc_block_idx + hd_offsets

        if mmc_token_idx == 0:
            # clear stored value with current key value
            tl.store(mmc_max_ptr, key)
            tl.store(mmc_min_ptr, key)
        else:
            # update stored value
            tl.store(mmc_max_ptr, tl.maximum(key, tl.load(mmc_max_ptr)))
            tl.store(mmc_min_ptr, tl.maximum(key, tl.load(mmc_min_ptr)))

        # begin the actual HSA kernel
        # first check if we can skip HSA kernel for small block sizes
        seq_len = tl.load(seq_lens_ptr)
        num_vllm_blocks = (seq_len - 1) // VLLM_BLOCK_SIZE + 1 # cdiv
        num_pool_blocks = (seq_len - 1) // POOL_BLOCK_SIZE + 1 # cdiv

        if num_pool_blocks <= TOP_K_1 and False:
            k1_ind = tl.arange(0, TOP_K_1)

            # all blocks fit in budget — write same IDs for every KV group
            log_pool_block_tok = POOL_BLOCK_SIZE * k1_ind

            log_vllm_block_ids = log_pool_block_tok // VLLM_BLOCK_SIZE

            phys_vllm_block_ids = tl.load(block_table_ptr + log_vllm_block_ids)

            phys_pool_block_ids = VLLM_BLOCK_SIZE * phys_vllm_block_ids + log_pool_block_tok % VLLM_BLOCK_SIZE 
            phys_pool_block_ids = phys_pool_block_ids // POOL_BLOCK_SIZE

            phys_pool_block_ids = tl.where(k1_ind < num_pool_blocks, phys_pool_block_ids, 0)

            tl.store(
                scratch_table_ptr + TOP_K_1 * head_idx + k1_ind,
                phys_pool_block_ids
            )

            return

        # Per-KV-group scoring with sum-of-softmax top-K selection
        # Output format: scratch_table row = (NUM_KV_GROUPS, MAX_BLOCK_BUDGET) flattened
        #
        # Instead of averaging queries and doing top-k on raw logits, we:
        #   1. Score each query independently against each page's min/max
        #   2. Maintain per-query raw scores in the top-k buffer
        #   3. At each merge step, recompute sum-of-softmax over the full
        #      buffer + new chunk, then select top-k based on that score.
        # This approximates true top-k of sum-of-softmax because the softmax
        # denominator changes as new pages are seen, so we recalculate it
        # each time the candidate set changes.

        num_topk_blocks = (num_pool_blocks - 1) // TOPK_BLOCK_SIZE + 1 # cdiv

        pos_inf = 3e5
        neg_inf = -3e5


        # create summary query 

        q_group_offsets = head_idx * NUM_Q_PER_HEAD + tl.arange(0, NUM_Q_PER_HEAD)
        q_hd_offsets = tl.arange(0, HIDDEN_DIM)

        raw_query_ptr = query_ptr + query_stride_0 * q_start + HIDDEN_DIM * q_group_offsets[:, None] + q_hd_offsets[None, :]

        # [H_q, h_d]
        queries = tl.load(raw_query_ptr)

        q_summary = tl.sum(queries / NUM_Q_PER_HEAD, axis=0)
        mmc_pick_max = q_summary > 0

        # use for online softmax
        logit_max = tl.full((NUM_Q_PER_HEAD, 1), value=float("-inf"), dtype=tl.float32)
        logit_expsum = tl.full((NUM_Q_PER_HEAD, 1), value=0, dtype=tl.float32)


        # Initialize running top-k buffer
        running_topk_ids = tl.full((TOP_K_2,), value=0, dtype=tl.int32)
        # Post-softmax scores: [MAX_BLOCK_BUDGET, NUM_Q_PER_HEAD]
        running_topk_exp = tl.full(
            (NUM_Q_PER_HEAD, TOP_K_2), 
            value=0, 
            dtype=tl.float32
        )

        running_buf_size = 0

        for i in range(num_topk_blocks):
            log_pool_block_segm = TOPK_BLOCK_SIZE * i + tl.arange(0, TOPK_BLOCK_SIZE)
            log_pool_block_mask = (log_pool_block_segm < num_pool_blocks)

            log_pool_block_toks = log_pool_block_segm * POOL_BLOCK_SIZE
            log_vllm_block_segm = log_pool_block_toks // VLLM_BLOCK_SIZE

            phys_vllm_block_ids = tl.load(
                block_table_ptr + log_vllm_block_segm,
                mask=log_pool_block_mask,
                other=0
            )

            phys_pool_block_ids = (phys_vllm_block_ids * VLLM_BLOCK_SIZE + log_pool_block_toks % VLLM_BLOCK_SIZE) // POOL_BLOCK_SIZE
            

            mmc_block_ids = kv_top - phys_pool_block_ids


            mmc_max_ptr_l = key_cache_ptr   + kv_stride_1 * mmc_block_ids[:, None] + hd_offsets[None, :]
            mmc_min_ptr_l = value_cache_ptr + kv_stride_1 * mmc_block_ids[:, None] + hd_offsets[None, :]

            mmc_block_max_l = tl.load(mmc_max_ptr_l, mask=log_pool_block_mask[:, None], other=0)
            mmc_block_min_l = tl.load(mmc_min_ptr_l, mask=log_pool_block_mask[:, None], other=0)

            # [TOPK_BLOCK_SIZE, HIDDEN_DIM]
            mmc_block_bounds = tl.where(mmc_pick_max, mmc_block_max_l, mmc_block_min_l)
            mmc_block_bounds = tl.trans(mmc_block_bounds)

            # [NUM_Q_PER_HEAD, TOP_K_BLOCK_SIZE]
            qk_hat = tl.dot(queries, mmc_block_bounds)


            qk_hat = tl.where(log_pool_block_mask[None, :], qk_hat, float("-inf"))

            # merge scaling for the new tile into softmax calculation. 
            # this way, we don't need to scale later
            qk_logit_max = tl.max(qk_hat, axis=1, keep_dims=True)
            new_logit_max = tl.maximum(qk_logit_max, logit_max)

            qk_logit_exp = tl.exp(qk_hat - new_logit_max)
            qk_logit_expsum = tl.sum(qk_logit_exp, axis=1, keep_dims=True)


            scale_factor = tl.exp(logit_max - new_logit_max)
            running_topk_exp = scale_factor * running_topk_exp

            logit_expsum = scale_factor * logit_expsum + qk_logit_expsum

            buffer_scores = running_topk_exp / logit_expsum
            buffer_scores = tl.sum(buffer_scores, axis=0)
            buffer_scores = tl.where(tl.arange(0, TOP_K_2) < running_buf_size, buffer_scores, neg_inf)

            incoming_scores = qk_logit_exp / logit_expsum
            incoming_scores = tl.sum(incoming_scores, axis=0)

            tl.store(
                debug_ptr + TOPK_BLOCK_SIZE*HIDDEN_DIM*i + tl.arange(0, TOPK_BLOCK_SIZE)[:, None] * HIDDEN_DIM + tl.arange(0, HIDDEN_DIM),
                mmc_block_max_l
            )

            # force last block selection, we want it to come first
            incoming_scores = tl.where(log_pool_block_segm == num_pool_blocks - 1, pos_inf, incoming_scores)
            incoming_scores = tl.where(log_pool_block_mask, incoming_scores, neg_inf)


            # top-k merge: iteratively pick the global max from buffer or incoming.
            # triton lacks sorting, so we use repeated argmax + masking.
            updated_ids = running_topk_ids * 0      # (TOP_K_2,) int32 zeros
            updated_exp = running_topk_exp * 0.0    # (NUM_Q_PER_HEAD, TOP_K_2) fp32 zeros

            for k in range(TOP_K_2):
                buf_max = tl.max(buffer_scores, axis=0)       # scalar
                buf_idx = tl.argmax(buffer_scores, axis=0)    # scalar
                inc_max = tl.max(incoming_scores, axis=0)     # scalar
                inc_idx = tl.argmax(incoming_scores, axis=0)  # scalar

                out_mask = (tl.arange(0, TOP_K_2) == k)  # (TOP_K_2,)

                if buf_max == neg_inf and inc_max == neg_inf:
                    pass  # both exhausted, remaining slots stay zero-filled
                elif buf_max > inc_max:
                    # pick from buffer
                    buf_mask = (tl.arange(0, TOP_K_2) == buf_idx)  # (TOP_K_2,)

                    phys_id  = tl.sum(tl.where(buf_mask, running_topk_ids, 0))               # scalar
                    sel_exp  = tl.sum(tl.where(buf_mask[None, :], running_topk_exp, 0.0),    # (NUM_Q_PER_HEAD, 1)
                                      axis=1, keep_dims=True)

                    updated_ids = tl.where(out_mask,         phys_id, updated_ids)           # (TOP_K_2,)
                    updated_exp = tl.where(out_mask[None, :], sel_exp, updated_exp)          # (NUM_Q_PER_HEAD, TOP_K_2)
                    buffer_scores = tl.where(buf_mask, neg_inf, buffer_scores)               # (TOP_K_2,)
                else:
                    # pick from incoming
                    inc_mask = (tl.arange(0, TOPK_BLOCK_SIZE) == inc_idx)  # (TOPK_BLOCK_SIZE,)

                    phys_id  = tl.sum(tl.where(inc_mask, phys_pool_block_ids, 0))            # scalar
                    sel_exp  = tl.sum(tl.where(inc_mask[None, :], qk_logit_exp, 0.0),        # (NUM_Q_PER_HEAD, 1)
                                      axis=1, keep_dims=True)

                    updated_ids = tl.where(out_mask,          phys_id, updated_ids)          # (TOP_K_2,)
                    updated_exp = tl.where(out_mask[None, :], sel_exp, updated_exp)          # (NUM_Q_PER_HEAD, TOP_K_2)
                    incoming_scores = tl.where(inc_mask, neg_inf, incoming_scores)           # (TOPK_BLOCK_SIZE,)


            # update
            running_topk_ids = updated_ids
            running_topk_exp = updated_exp

            # TOPK_BLOCK_SIZE is not the true increment since some blocks can be empty at the end
            # however, the loop won't re-execute after that so we are good
            running_buf_size = min(
                running_buf_size + TOPK_BLOCK_SIZE, 
                TOP_K_2
            )

        # we now have min(k_2, num_pool_blocks) inserted into buffer
        # we only want the k_1 top indices, or the number of pool blocks, whichever is lower
        actual_block_budget = min(TOP_K_1, num_pool_blocks)

        # reverses the offset
        flush_k_offset = tl.arange(0, TOP_K_2)
        flush_k_offset = tl.where(
            flush_k_offset < actual_block_budget,
            actual_block_budget - flush_k_offset - 1,
            flush_k_offset
        )

        tl.store(
            scratch_table_ptr + TOP_K_1 * head_idx + flush_k_offset,
            running_topk_ids,
            mask=flush_k_offset < actual_block_budget
        )


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
    pool_block_size : int,
    topk_token_budget : int

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
    kv_top          = key_cache.shape[0] * key_cache.shape[1] - 1
    VLLM_BLOCK_SIZE = key_cache.shape[1]
    NUM_KV_GROUPS   = key_cache.shape[2]
    HIDDEN_DIM      = key_cache.shape[3]

    # query: [num_tokens, num_kv_groups * num_q_per_group, hidden_dim]
    NUM_Q_PER_HEAD = query.shape[1] // NUM_KV_GROUPS

    # TOPK_BLOCK_SIZE controls how many KV pages we score per inner loop iteration.
    # 16 is a reasonable default: large enough to amortize loop overhead and
    # keep the tl.dot tile non-trivial, small enough to stay in registers.
    TOPK_BLOCK_SIZE = 16

    assert VLLM_BLOCK_SIZE % pool_block_size == 0
    assert topk_token_budget % pool_block_size == 0

    top_k1 = topk_token_budget // pool_block_size
    top_k2 = top_k1 * 2 # fix for now

    # prefix sum increases dim by 1, so we need to subtract 1
    num_reqs   = query_start_loc.shape[0] - 1
    num_heads = NUM_KV_GROUPS
    num_blocks = 128
    grid = (num_reqs, num_heads, num_blocks)

    debug = torch.zeros((num_reqs, 128, 16, 128), dtype=torch.bfloat16, device=key_cache.device)

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
        TOP_K_1                 = top_k1,
        TOP_K_2                 = top_k2,
        NUM_Q_PER_HEAD          = NUM_Q_PER_HEAD,
        HIDDEN_DIM              = HIDDEN_DIM,
        VLLM_BLOCK_SIZE         = VLLM_BLOCK_SIZE,
        POOL_BLOCK_SIZE         = pool_block_size,
        TOPK_BLOCK_SIZE         = TOPK_BLOCK_SIZE,
        debug_ptr=debug,
        debug_stride_0=debug.stride(0)
    )

    #print("Dbg:")
    #print(debug[:, 0])

def consolidate_per_head_selection(
    key_cache : torch.Tensor,
    value_cache : torch.Tensor,
    block_table : torch.Tensor,
    scratch_table : torch.Tensor,
    query_start_loc : torch.Tensor,
    seq_lens : torch.Tensor,
    pool_block_size : int,
    topk_token_budget : int,
    streaming_llm : bool = False,
    num_sink_tokens : int = 4,
):
    """Consolidate per-KV-head selections into a single block table mapping.

    Must be called after select_top_pages(). Permutes the KV cache in-place
    so that Flash Attention (which uses a single block table for all heads)
    sees the correct data for each KV head.

    If streaming_llm=True, ignores the per-head HSA selections from
    scratch_table and instead uses the StreamingLLM selection strategy:
    the first num_sink_tokens (attention sinks) and the most recent tokens
    up to the remaining budget.  All KV heads share the same selection.
    """
    # key_cache: [num_blocks, block_size, num_kv_groups, hidden_dim]
    num_reqs = query_start_loc.shape[0] - 1
    VLLM_BLOCK_SIZE = key_cache.shape[1]
    NUM_KV_GROUPS = key_cache.shape[2]
    TOP_K_1 = topk_token_budget // pool_block_size

    kv_top = key_cache.shape[0] * VLLM_BLOCK_SIZE - 1

    raw_key_cache = key_cache

    key_cache = key_cache.flatten(0, 1).unflatten(0, (-1, pool_block_size))
    value_cache = value_cache.flatten(0, 1).unflatten(0, (-1, pool_block_size))



    query_len = query_start_loc.diff()
    decode_req = (query_len == 1)

    if streaming_llm:
        # Build source_ids from StreamingLLM position-based selection.
        #
        # Selection strategy (per decode request):
        #   - Sink pages  : logical pool blocks 0 .. num_sink_pages-1
        #   - Recent pages: logical pool blocks (num_pool_blocks - num_recent_pages) .. (num_pool_blocks - 1)
        #
        # When a sequence fits entirely within the budget we fall back to the
        # canonical mapping (swap is a no-op) so we never corrupt short seqs.
        # All KV heads share the same selection (StreamingLLM is head-agnostic).

        num_sink_pages = (num_sink_tokens + pool_block_size - 1) // pool_block_size
        num_recent_pages = TOP_K_1 - num_sink_pages
        num_pool_blocks = (seq_lens.long() + pool_block_size - 1) // pool_block_size  # [num_reqs]

        # Sink logical indices: same for every request
        sink_idx = torch.arange(num_sink_pages, device=block_table.device, dtype=torch.long)
        sink_idx = sink_idx.unsqueeze(0).expand(num_reqs, -1)  # [num_reqs, num_sink_pages]

        # Recent logical indices: last num_recent_pages pool blocks, non-overlapping with sinks
        recent_start = (num_pool_blocks - num_recent_pages).clamp(min=num_sink_pages)  # [num_reqs]
        recent_offset = torch.arange(num_recent_pages, device=block_table.device, dtype=torch.long)
        recent_idx = (recent_start[:, None] + recent_offset[None, :]).clamp(
            max=block_table.shape[1] - 1
        )  # [num_reqs, num_recent_pages]

        # Combined logical pool-block indices: [num_reqs, MBB]
        streaming_logical = torch.cat([sink_idx, recent_idx], dim=1)

        vllm_block_ind = (streaming_logical * pool_block_size) // VLLM_BLOCK_SIZE

        # Map to physical block IDs via block_table: [num_reqs, MBB]
        source_phys = block_table.gather(1, vllm_block_ind)

        streaming_phys = source_phys * VLLM_BLOCK_SIZE + (streaming_logical * pool_block_size) % VLLM_BLOCK_SIZE
        streaming_phys = streaming_phys // pool_block_size
        # delete overflow using logical indices
        streaming_phys = torch.where(streaming_logical < num_pool_blocks.unsqueeze(1), streaming_phys, 0) 

        # For short sequences (fit in budget) or non-decode requests, use the
        # canonical positions directly so the swap becomes a no-op.
        source_phys = torch.where(decode_req.unsqueeze(1), streaming_phys, 0)
        source_ids = source_phys.flatten()

        source_ids = streaming_phys

        # Expand the same selection to every KV group then reshape to match
        # the format expected by the gather/scatter block below.
        # [num_reqs, NUM_KV_GROUPS, MBB] -> [num_reqs * MBB, NUM_KV_GROUPS]
        #source_ids = source_phys.unsqueeze(1).expand(-1, NUM_KV_GROUPS, -1)
        #source_ids = source_ids.transpose(1, 2).flatten(0, 1)
        #source_ids = source_ids[:, None, :, None]  # [num_reqs * MBB, 1, NUM_KV_GROUPS, 1]

    else:
        source_ids = scratch_table[:, :NUM_KV_GROUPS * TOP_K_1].unflatten(
            1, (NUM_KV_GROUPS, TOP_K_1)
        )


        # pointing to zero page will nullify the op
        # [num reqs, kv groups, mbb]
        source_ids = torch.where(decode_req[:, None, None], source_ids, 0)

        #print(torch.sort(source_ids, descending=True).values[:, :, :20])
        #print("Which has block table")
        #print((torch.sort(block_table, descending=True).values * VLLM_BLOCK_SIZE // pool_block_size)[:, :20])

        #print(source_ids[:, :, -20:])
        #print("Which has block table:")
        #print(block_table[:, :20])

        # we do not make a distinction between requests in KV group
        # we can merge the entire operation into a non-batched tensor
        # now the second dim has all source IDs across all requests
        # [num blocks, kv groups]
        source_ids = source_ids.transpose(1, 2).flatten(0, 1)

        # get to appropriate format for gathering from KV cache
        # kv cache size is [num blocks, block size, kv groups, hd]
        # obtain size [num blocks, block size, kv groups, hd]
        source_ids = source_ids[:, None, :, None]

    # for vllm block size = pool block size, we can just reuse existing canonical ids
    # duplicate canonical ids across all source IDs
    # [num blocks, 1, kv groups, 1]
    # canonical_ids = source_ids[:, :, :1].expand_as(source_ids)
    # if that is not the case, we need to come up with custom canonical ids, 
    # both for pool block indices and the scratch table itself (since scratch table accepts block sizes only of vllm block size)
    # there's actually a few ways to go about this:
    # 1) fork the attention kernels to support smaller block sizes. 
    #    this is what we need to do anyways to have zero-copy per-KV paging
    #    however, kernel modification carries its own risks and ask noted by the vLLM paper, small page sizes are more inefficient
    # 2) copy all selected tokens to fixed-ahead-of-time positions in the KV cache. 
    #    the most canonical way to do this is to select the first n pages as our target
    #    this is destructive to the KV cache because we move the last HSA page, which will prevent us from adding more tokens in the future
    # 3) copy all selected tokens to dynamic positions. while this does fix the previous approach's problems, we end up not reaching our token
    #    budget exactly since the last page shouldn't be modified beyond the current token
    # 4) copy to a new scratch buffer. this is the simplest approach, and does not require items to be moved out of the way. 
    #    we can extend this with a set-associative cache to reduce memcpys to nearly zero in most cases
    #    however, this becomes problematic if we want to combine prefill and decode in the same request - how do we fit an arbitrary context in a short scratch buffer?
    #    furthermore, if we want a set-associative cache, we are going to have massive memory redundnacy that is going to waste memory
    #    one crazy hacky solution is to allow to scratch buffer to index into the original kv cache by jumping across address space 
    #    this will allow access of the original data, but then what if the index is negative and the kernel supports only unsigned indices?
    # 5) hybrid solution. we maintain a new block to copy the last vllm page size / block size tokens, and leave everything else in-place in the kv cache. 
    #    we utilize a set-associative cache to prevent unncessary copies. 
    #    this still leaves the question of top-k for very large token budgets and small block size
    #    for top-k size 2048, we have 1024 top-k slots, each one requires 4 bytes int id + 8 byte fp 16 exponential
    #    12k memory, this adds up    



    num_vllm_blocks = topk_token_budget // VLLM_BLOCK_SIZE

    dummy_block = raw_key_cache[:1]

    key_cache[0].fill_(torch.nan)
    value_cache[0].fill_(torch.nan)
    
    #print(f"Source IDS: {source_ids.view(num_reqs, -1).shape}\n{source_ids.view(num_reqs, -1)}")

    k2 = key_cache[source_ids].view(-1, *raw_key_cache.shape[1:])
    v2 = value_cache[source_ids].view(-1, *raw_key_cache.shape[1:])

    old_k2 = k2

    k2 = torch.cat([dummy_block, k2], dim=0)
    v2 = torch.cat([dummy_block, v2], dim=0)

    bt2 = torch.arange(0, num_reqs * num_vllm_blocks, device=block_table.device, dtype=torch.int32) + 1
    bt2 = bt2.view(num_reqs, num_vllm_blocks)

    actual_blocks = (seq_lens - 1) // VLLM_BLOCK_SIZE + 1

    bt2 = torch.where(torch.arange(0, num_vllm_blocks, device=block_table.device, dtype=torch.int32)[None, :] < actual_blocks[:, None], bt2, 0)

    return k2, v2, bt2

    raise RuntimeError("unreachable point")

    # to calculate canonical ids, just use the first few blocks
    # first, obtain the logical indices for canonical ids, then convert to physical 
    canonical_offs = torch.arange(
        start=0, 
        end=topk_token_budget, 
        step=pool_block_size, 
        device=source_ids.device
    ) % VLLM_BLOCK_SIZE
    canonical_offs = canonical_offs.view(num_vllm_blocks, VLLM_BLOCK_SIZE // pool_block_size)

    # convert block table to physical token locations, then add to offsets
    # broadcast across all reqs
    canonical_ids = block_table[:, :num_vllm_blocks, None] * VLLM_BLOCK_SIZE + canonical_offs[None, :]

    # flatten to get back flat array of token positions for each req
    # shape ends up being [num reqs, total block budget]
    canonical_ids = canonical_ids.flatten(1, 2)

    # convert from absolute token positions to to pool block page IDs
    canonical_ids = canonical_ids // pool_block_size


    # zero out prefill requests
    canonical_ids = torch.where(decode_req[:, None], canonical_ids, 0)

    # [num reqs * total block budget]
    canonical_ids = canonical_ids.flatten()
    # broad castto [num reqs * total block budget, pages per block, kv groups, h_d]
    canonical_ids = canonical_ids[:, None, None, None].expand_as(source_ids)


    # collect canonical and source pages from table
    # [num blocks, block size, kv groups, hd]
    src_keys = key_cache.gather(0, source_ids)
    can_keys = key_cache.gather(0, canonical_ids)

    key_cache.scatter_(0, canonical_ids, src_keys)
    #key_cache.scatter_(0, source_ids, can_keys)

    src_vals = value_cache.gather(0, source_ids)
    can_vals = value_cache.gather(0, canonical_ids)

    value_cache.scatter_(0, canonical_ids, src_vals)
    #value_cache.scatter_(0, source_ids, can_vals)

    # mm cache has shape [num toks, kv groups, hd]
    # target shape: [num blocks, kv groups, hd]
    mmc_src_ids = kv_top - source_ids.squeeze(1)
    mmc_can_ids = kv_top - canonical_ids.squeeze(1)

    assert key_cache.is_contiguous() and value_cache.is_contiguous()
    max_cache = key_cache.flatten(0, 1)
    min_cache = value_cache.flatten(0, 1)

    src_max_keys = max_cache.gather(0, mmc_src_ids)
    can_max_keys = max_cache.gather(0, mmc_can_ids)

    max_cache.scatter_(0, mmc_can_ids, src_max_keys)
    max_cache.scatter_(0, mmc_src_ids, can_max_keys)

    src_min_keys = min_cache.gather(0, mmc_src_ids)
    can_min_keys = min_cache.gather(0, mmc_can_ids)

    min_cache.scatter_(0, mmc_can_ids, src_min_keys)
    min_cache.scatter_(0, mmc_src_ids, can_min_keys)

    scratch_table.copy_(block_table)
