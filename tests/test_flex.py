from pathlib import Path

import fire
import torch.cuda
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from finetune.args import TrainArgs
from finetune.data.data_loader import build_data_loader
from finetune.wrapped_model import load_args
from model.transformer import Transformer, generate_prefix_packed_mask_mod, get_prefix_lens
import torch
from torch.nn.attention.flex_attention import create_block_mask
import torch
from typing import Optional
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import math
from torch.nn.attention.flex_attention import (
    _score_mod_signature,
    _mask_mod_signature,
    _vmap_for_bhqkv,
    _ModificationType,
)
from torch._higher_order_ops.flex_attention import TransformGetItemToIndex
from contextlib import nullcontext
Tensor = torch.Tensor

def main(config):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)

    if Path(args.model_id_or_path).is_dir():
        model_folder = Path(args.model_id_or_path)
    else:
        raise ValueError(
            "Invalid folder path. Please set `args.initial_model` to a valid folder path."
        )

    # 6. Load function calling instruct tokenizer
    vocab_size = load_args(model_folder, args.lora).vocab_size
    is_tekken = vocab_size > 32768

    if not is_tekken:
        tokenizer_path = "/home/jonathan/research/project/mistral-common/src/tokenizer_new.model.v3"
        instruct_tokenizer = MistralTokenizer.from_file(tokenizer_path).instruct_tokenizer # type: ignore
    else:
        instruct_tokenizer = MistralTokenizer.v3(
            is_tekken=is_tekken
        ).instruct_tokenizer  # type: ignore

    print("Loading data")
    data_loader = build_data_loader(
        instruct_tokenizer=instruct_tokenizer,
        args=args.data,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=0,  # DDP rank
        world_size=1,  # DDP world_size
        is_eval=False,
    )
    batch = next(data_loader)
    x = torch.from_numpy(batch.x).cuda(non_blocking=True)
    seqlens = batch.sizes
    print("Finish data")

    # assert args.lora is not None, "`args.lora` should be set to a valid value."
    # model_args = load_args(model_folder, args.lora)
    print("Loading model")
    # model = Transformer(
    #     args=model_args,
    #     checkpoint=args.checkpoint,
    # )
    # print("Finish model")
    # mask = model._get_masks(x, seqlens)
    # breakpoint()
    device='cuda'
    seq_ids = torch.tensor([seq_id for seq_id, length in enumerate(seqlens) for _ in range(length)], device=x.device)
    prefix_lens = get_prefix_lens(x, seqlens, seq_ids)
    mask_mod = generate_prefix_packed_mask_mod(seq_ids, prefix_lens)
    print(create_block_mask(mask_mod, 1, 1, 16384, 16384, device=device))


    def make_tensor():
        return torch.ones(1, 1, 16384, 64, device=device)

    query, key = make_tensor(), make_tensor()

    def _name_to_title(name: str) -> str:
        title = name.replace("_", " ")
        title = " ".join(word.capitalize() for word in title.split())
        return title


    def create_score_mod(
        query: torch.Tensor,
        key: torch.Tensor,
        score_mod: Optional[_score_mod_signature],
        mask_mod: Optional[_mask_mod_signature],
        device: str = "cuda",
        _compile: bool = False,
        scale: Optional[float] = None,
        batch_idx: int = 0,
        head_idx: int = 0,
    ) -> torch.Tensor:
        B = 1
        H = 1
        M = query.shape[0]
        N = key.shape[0]

        b = torch.arange(0, B, device=device) + batch_idx
        h = torch.arange(0, H, device=device) + head_idx
        m = torch.arange(0, M, device=device)
        n = torch.arange(0, N, device=device)

        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        type = _ModificationType.SCORE_MOD if score_mod is not None else _ModificationType.MASK_MOD
        if _compile:
            ctx = nullcontext()
        else:
            ctx = TransformGetItemToIndex()

        with ctx:
            mod_fn = score_mod if type == _ModificationType.SCORE_MOD else mask_mod
            prefix = (0,) if type == _ModificationType.SCORE_MOD else ()
            mod = _vmap_for_bhqkv(mod_fn, prefix=prefix)
            scores = query @ key.transpose(-2, -1)
            scores *= scale_factor
            scores = scores.view(1, 1, M, N)
            if type == _ModificationType.SCORE_MOD:
                out = mod(scores, b, h, m, n)
            else:
                out = mod(b, h, m, n)

        return out

    def visualize_attention_scores(
        query: Tensor,
        key: Tensor,
        score_mod: Optional[_score_mod_signature] = None,
        mask_mod: Optional[_mask_mod_signature] = None,
        device: str = "cuda",
        name: str = "attention_scores",
        path: Optional[Path] = None,
        batch_idx: int = 0,
        head_idx: int = 0,
        scale: Optional[float] = None,
    ):
        """
        Generate and save a visualization of attention scores.

        Args:
            query (Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
            key (Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim).
            score_mod (Optional[Callable]): If this is set this will take precedence over the mask_mod.
            mask_mod (Optional[Callable]): The mask_mod function used to create block_mask
            device (str): Device to run computations on (default: "cuda").
            name (str): Base name for the file and title (default: 'attention_scores').
            path (Path): Path to save the visualization. If None, will be saved to the current working directory.
            batch_idx (int): Index of the batch to visualize (default: 0).
            head_idx (int): Index of the head to visualize (default: 0).
            scale (float): Scale factor to apply to the attention scores. If None, will be set to 1 / sqrt(head_dim).

        Returns:
            None
        """
        assert (
            score_mod is not None or mask_mod is not None
        ), "Must provide either score_mod or mask_mod"
        query = query[batch_idx, head_idx, :, :]
        key = key[batch_idx, head_idx, :, :]
        scores_viz = create_score_mod(
            query,
            key,
            score_mod=score_mod,
            mask_mod=mask_mod,
            scale=scale,
            device=device,
            batch_idx=batch_idx,
            head_idx=head_idx,
        )

        suffix_title = f"Batch {batch_idx}, Head {head_idx}" if batch_idx != 0 or head_idx != 0 else ""

        fig, ax = plt.subplots(figsize=(12, 10))
        color = "viridis" if score_mod is not None else "cividis"
        im = ax.imshow(scores_viz.cpu().detach()[0, 0, :, :], aspect="auto", cmap=color)
        fig.colorbar(im)

        title = _name_to_title(name)
        file_path = Path(name).with_suffix(".png") if path is None else path.with_suffix(".png")
        ax.set_title(f"{title}\n{suffix_title}", fontsize=20)

        ax.set_xlabel("Key Tokens", fontsize=18)
        ax.set_ylabel("Query Tokens", fontsize=18)

        # Move y-axis ticks and labels to the top
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

        # Add tick labels if the number of tokens is manageable
        num_query_tokens, num_kv_tokens = scores_viz.shape[-2:]
        if num_query_tokens <= 32 and num_kv_tokens <= 32:
            ax.set_xticks(range(num_kv_tokens))
            rotation = 45 if num_kv_tokens > 12 else 0
            ax.set_xticklabels(
                [f"KV{i}" for i in range(num_kv_tokens)], fontsize=16, rotation=rotation
            )
            ax.set_yticks(range(num_query_tokens))
            ax.set_yticklabels([f"Q{i}" for i in range(num_query_tokens)], fontsize=16)
            # Align grid with pixel boundaries
            ax.set_xticks(np.arange(-0.5, num_kv_tokens, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, num_query_tokens, 1), minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free up memory

        print(f"Visualization saved as {file_path}")

    visualize_attention_scores(query, key, mask_mod=mask_mod, device=device, name="causal_mask")

if __name__ == "__main__":
    fire.Fire(main)
