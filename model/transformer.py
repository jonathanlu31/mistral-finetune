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

def get_prefix_lens(input_ids, seqlens):
    END_SYS_TOKEN_ID = 11
    prefix_lens = torch.where(input_ids == END_SYS_TOKEN_ID)[0] + 1
    assert len(prefix_lens) == len(seqlens)
    return prefix_lens

def is_torch_nightly_installed():
    return 'dev' in torch.__version__ or torch.__version__.endswith('+git')

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
        inner_mask = kv_logical < prefix_lens[seq_ids[q_idx]] or q_logical >= kv_logical
        return same_doc & inner_mask
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

        if is_torch_nightly_installed():
            from torch.nn.attention.flex_attention import flex_attention
            flex_attention = torch.compile(flex_attention, dynamic=False)
            self.attn_op = lambda q,k,v,mask: flex_attention(q,k,v,block_mask=mask)
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
        if is_torch_nightly_installed():
            from torch.nn.attention.flex_attention import create_block_mask
            seq_ids = [seq_id for seq_id, length in enumerate(seqlens) for _ in range(length)]
            prefix_lens = get_prefix_lens(input_ids)
            mask_mod = generate_prefix_packed_mask_mod(seq_ids, prefix_lens)
            return create_block_mask(mask_mod, None, None, self.max_seq_len, self.max_seq_len)
        else:
            from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
            return BlockDiagonalCausalMask.from_seqlens(seqlens)


def positions_from_sizes(sizes: Iterable[int], device):
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )
