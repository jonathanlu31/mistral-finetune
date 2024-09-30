import operator
from functools import partial, reduce
from typing import Iterable, List, Optional, Union

import torch
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as torch_ckpt
import torch.nn as nn

from .args import ModelArgs
from .lora import LoRALinear
from .moe import MoeLayer
from .rope import apply_rotary_emb, precompute_freqs_cis

def is_torch_nightly_installed():
    return 'dev' in torch.__version__ or torch.__version__.endswith('+git')

_SUPPORTS_FLEX_ATTENTION = (
    is_torch_nightly_installed()
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (7, 5)
)

if _SUPPORTS_FLEX_ATTENTION:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask as create_block_causal_mask_flex,
        flex_attention,
    )

    flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

    # We cannot do nested compile, but flex attention only has perf benefits
    # when compiled. To insulate it from the compiler, we wrap it with
    # compiler.disable so that it can be used regardless of whether the model
    # is compiled or not, and flex attention always remains compiled.
    @torch.compiler.disable(recursive=False)
    def compile_friendly_flex_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        return flex_attention_compiled(q, k, v, block_mask=block_mask)

def get_prefix_lens(input_ids, seqlens, seq_ids):
    END_SYS_TOKEN_ID = 11
    end_of_sys_idxs = torch.where(input_ids == END_SYS_TOKEN_ID)[0] + 1
    sequences_with_end_sys = seq_ids[end_of_sys_idxs]

    # Start index of all sequences
    all_sequence_starts = torch.cumsum(torch.tensor([0] + seqlens[:-1], device=input_ids.device), dim=0)

    # prefix lens of sequences with a system message
    prefix_lens = end_of_sys_idxs - all_sequence_starts[sequences_with_end_sys]
    all_prefix_lens = torch.zeros(len(seqlens), device=input_ids.device, dtype=prefix_lens.dtype)
    all_prefix_lens[sequences_with_end_sys] = prefix_lens
    return all_prefix_lens

def generate_prefix_packed_mask_mod(seq_ids, prefix_lens):
    # Get unique document IDs and their counts
    _, counts = torch.unique_consecutive(seq_ids, return_counts=True)
    # Create cumulative counts (offsets)
    offsets = torch.cat([torch.tensor([0], device=seq_ids.device), counts.cumsum(0)[:-1]])
    def doc_mask_wrapper(_b, _h, q_idx, kv_idx):
        same_doc = seq_ids[q_idx] == seq_ids[kv_idx]
        q_logical = q_idx - offsets[seq_ids[q_idx]]
        kv_logical = kv_idx - offsets[seq_ids[kv_idx]]

        # prefix causal lm
        inner_mask = torch.logical_or(kv_logical < prefix_lens[seq_ids[q_idx]], q_logical >= kv_logical)
        return torch.logical_and(same_doc, inner_mask)
    return doc_mask_wrapper


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def maybe_lora_layer(
    args: ModelArgs, rank: Optional[int] = None
) -> Union[partial[LoRALinear], type[nn.Linear]]:
    MaybeLora: Union[partial[LoRALinear], type[nn.Linear]]
    if not args.lora.enable:
        return nn.Linear

    rank = rank or args.lora.rank
    scaling = args.lora.scaling
    dropout = args.lora.dropout

    MaybeLora = partial(
        LoRALinear,
        rank=rank,
        scaling=scaling,
        dropout=dropout,
    )

    return MaybeLora


class Attention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads
        self.head_dim: int = args.head_dim

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        MaybeLora = maybe_lora_layer(args)

        self.wq = MaybeLora(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = MaybeLora(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = MaybeLora(args.dim, args.n_kv_heads * args.head_dim, bias=False)

        self.wo = MaybeLora(args.n_heads * args.head_dim, args.dim, bias=False)

        if _SUPPORTS_FLEX_ATTENTION:
            self.attn_op = lambda q,k,v,mask: compile_friendly_flex_attention(q,k,v,block_mask=mask)
        else:
            from xformers.ops.fmha import memory_efficient_attention
            self.attn_op = lambda q,k,v,mask: memory_efficient_attention(q, k, v, mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask,
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(seqlen_sum, self.n_heads, self.args.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        key, val = xk, xv

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        
        output = self.attn_op(xq, key, val, mask=mask)

        return self.wo(output.view(seqlen_sum, -1))


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        MaybeLora = maybe_lora_layer(args)
        self.w1 = MaybeLora(args.dim, args.hidden_dim, bias=False)
        self.w2 = MaybeLora(args.hidden_dim, args.dim, bias=False)
        self.w3 = MaybeLora(args.dim, args.hidden_dim, bias=False)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)

        self.feed_forward: Union[MoeLayer, FeedForward]
        if args.moe is not None:
            self.feed_forward = MoeLayer(
                experts=[FeedForward(args=args) for _ in range(args.moe.num_experts)],
                gate=nn.Linear(args.dim, args.moe.num_experts, bias=False),
                moe_args=args.moe,
            )
        else:
            self.feed_forward = FeedForward(args=args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        att_mask,
    ) -> torch.Tensor:
        r = self.attention(self.attention_norm(x), freqs_cis, att_mask)
        h = x + r

        r = self.feed_forward(self.ffn_norm(h))
        out = h + r

        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, checkpoint: bool = False):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for _ in range(args.n_layers):
            block: torch.nn.Module = TransformerBlock(args=args)
            if checkpoint:
                # activate gradient checkpointing as, see: https://pytorch.org/docs/stable/checkpoint.html
                non_reentrant_wrapper = partial(
                    torch_ckpt.checkpoint_wrapper,
                    checkpoint_impl=torch_ckpt.CheckpointImpl.NO_REENTRANT,
                )
                block = non_reentrant_wrapper(block)

            self.layers.append(block)

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.output = torch.nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False,
        )

        # set lazily
        self._freqs_cis = None

    @property
    def dtype(self) -> torch.dtype:
        return self.tok_embeddings.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.tok_embeddings.weight.device

    @property
    def freqs_cis(self):
        # lazy init
        device = next(iter(self.parameters())).device
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000, theta=self.args.rope_theta, device=device
            )

        return self._freqs_cis

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
    ) -> torch.Tensor:
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])

        h = self.tok_embeddings(input_ids)
        positions = positions_from_sizes(seqlens, self.freqs_cis.device)
        att_mask = self._get_masks(input_ids, seqlens)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        for layer in self.layers:
            h = layer(h, freqs_cis, att_mask)

        return self.output(self.norm(h)).float()

    def _get_masks(self, input_ids, seqlens):
        if _SUPPORTS_FLEX_ATTENTION:
            from torch.nn.attention.flex_attention import create_block_mask
            seq_ids = torch.tensor([seq_id for seq_id, length in enumerate(seqlens) for _ in range(length)], device=input_ids.device)
            prefix_lens = get_prefix_lens(input_ids, seqlens, seq_ids)
            mask_mod = generate_prefix_packed_mask_mod(seq_ids, prefix_lens)
            # TODO: Fix hard code
            return create_block_mask(mask_mod, None, None, 8192, 8192)
        else:
            from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
            return BlockDiagonalCausalMask.from_seqlens(seqlens)


def positions_from_sizes(sizes: Iterable[int], device):
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )
