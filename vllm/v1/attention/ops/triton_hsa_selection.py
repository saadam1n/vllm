import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


# warmup example
@triton.jit
def hsa_update_page_minmax(
    query_start_loc_ptr, # [batch + 1,]
    slot_mapping_ptr, # [number of queries,]
    key_ptr, # [number of queries, number of KV groups, hidden dim]
    key_cache_ptr, # [number of blocks, block size, number of KV groups, hidden dim]
    value_cache_ptr, # [number of blocks, block size, number of KV groups, hidden dim]
    key_stride_0 : int, # stride of key_ptr dim 1 (needed in some cases)
    kv_stride_1 : int, # corresponds to stride of number of KV groups * hidden dim * sizeof(dtype)
    kv_top : int, # token with highest index in KV cache
    FULL_KEY_DIM : tl.constexpr, # number of KV groups * hidden dim 
    BLOCK_SIZE : tl.constexpr
):
    # launch dims:
    # [batch, blocks]
    batch_idx = tl.program_id(axis=0)
    block_idx = tl.program_id(axis=1)

    # get start and end positions as tensor of size (2,)
    q_start = tl.load(query_start_loc_ptr + batch_idx)
    q_end = tl.load(query_start_loc_ptr + batch_idx + 1)

    q_len = q_end - q_start

    if q_len > 1:
        # prefill
        # notably, we assume there is no case of spec decode
        # that is, each block begins at the start of the block (no updates, only clears)
        
        # each prefill handles one block
        # this may use a massive amount of smem
        # hopefully the compiler can figure it out
        
        q_base = q_start + BLOCK_SIZE * block_idx

        # iterate through remaining blocks
        while q_base < q_end:

            off_tokens = tl.arange(0, BLOCK_SIZE)


            off_key_dim = tl.arange(0, FULL_KEY_DIM)

            # (BLOCK_SIZE, 1)
            off_tokens = off_tokens[:, None]
            # (1, FULL_KEY_DIM)
            off_key_dim = off_key_dim[None, :]

            ld_mask = (q_base + off_tokens < q_end)

            keys = tl.load(key_ptr + key_stride_0 * (q_base + off_tokens) + off_key_dim, mask=ld_mask, other=float("-inf"))

            slot_idx = tl.load(slot_mapping_ptr + q_base)
            phys_block_idx = slot_idx // BLOCK_SIZE
            # assume sub_block_idx == 0 (see decode for more details)
            cache_slot_idx = kv_top - phys_block_idx 

            cache_max_ptr = key_cache_ptr   + kv_stride_1 * cache_slot_idx
            cache_min_ptr = value_cache_ptr + kv_stride_1 * cache_slot_idx

            tl.store(cache_max_ptr + tl.arange(0, FULL_KEY_DIM), tl.max(keys, axis=0))

            keys = tl.where(ld_mask, keys, float("inf"))
            tl.store(cache_min_ptr + tl.arange(0, FULL_KEY_DIM), tl.min(keys, axis=0))

            q_base += BLOCK_SIZE * tl.num_programs(axis=1) 
    else:
        # only run for block idx 0
        if block_idx != 0:
            return

        # decode
        key =  tl.load(key_ptr + key_stride_0 * q_start + tl.arange(0, FULL_KEY_DIM))

        slot_idx = tl.load(slot_mapping_ptr + q_start)

        # replace with bitshift and mask ops when compiling (hopefully)
        # I also need to update my naming conventions
        phys_block_idx = slot_idx // BLOCK_SIZE
        sub_block_idx = slot_idx % BLOCK_SIZE

        # note that block_idx cannot equal zero
        cache_slot_idx = kv_top - phys_block_idx 

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


def update_page_minmax(
    query_start_loc : torch.Tensor,
    slot_mapping : torch.Tensor,
    key : torch.Tensor,
    key_cache : torch.Tensor,
    value_cache : torch.Tensor
):
    assert query_start_loc.is_contiguous(), "HSA kernels currently have limited support for non-contiguous tensors"
    assert slot_mapping.is_contiguous(), "HSA kernels currently have limited support for non-contiguous tensors"
    # key is weird for some reason, so we check an indiviudal key rather than the whole thing
    assert key[0].is_contiguous(), "HSA kernels currently have limited support for non-contiguous tensors"
    assert key_cache.is_contiguous(), "HSA kernels currently have limited support for non-contiguous tensors"
    assert value_cache.is_contiguous(), "HSA kernels currently have limited support for non-contiguous tensors"


    kv_top = key_cache.shape[0] * key_cache.shape[1]
    FULL_KEY_DIM = key_cache.shape[2] * key_cache.shape[3]
    BLOCK_SIZE = key_cache.shape[1]

    # prefix sum increases dim by 1, so we need to subtract 1
    num_reqs = query_start_loc.shape[0] - 1
    num_blocks = 16
    grid = (num_reqs, num_blocks)

    hsa_update_page_minmax[grid](
        query_start_loc_ptr=query_start_loc,
        slot_mapping_ptr=slot_mapping,
        key_ptr=key,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        key_stride_0=key.stride(0),
        kv_stride_1=key_cache.stride(1),
        kv_top=kv_top,
        FULL_KEY_DIM=FULL_KEY_DIM,
        BLOCK_SIZE=BLOCK_SIZE
    )


"""
need to complete the actual kernel later
@triton.jit
def hsa_select(
    sel_blocks_ptr, 
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    seq_lens_ptr,  # [num_seqs]
):
    pass
"""