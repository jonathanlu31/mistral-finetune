from pathlib import Path
from typing import TYPE_CHECKING

import fire
import torch.cuda
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from finetune.args import TrainArgs
from finetune.data.data_loader import build_data_loader
from finetune.wrapped_model import load_args

from rich.console import Console
from rich.text import Text

if TYPE_CHECKING:
    from mistral_common.tokens.tokenizers.sentencepiece import InstructTokenizerBase

def main(config):
    console = Console()

    def print_colored(text, color):
        colored_text = Text(text, style=color)
        console.print(colored_text, end=", ")

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
        tokenizer_path = "/home/jonathan_lu/research/project/mistral-common/src/tokenizer_new.model.v3"
        instruct_tokenizer = MistralTokenizer.from_file(tokenizer_path).instruct_tokenizer # type: ignore
    else:
        instruct_tokenizer: InstructTokenizerBase = MistralTokenizer.v3(
            is_tekken=is_tekken
        ).instruct_tokenizer  # type: ignore

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
    y = torch.from_numpy(batch.y).cuda(non_blocking=True)
    y_mask = (
        torch.from_numpy(batch.y_mask).cuda(non_blocking=True)
        if batch.y_mask is not None
        else None
    )

    tokens = [instruct_tokenizer.tokenizer.id_to_piece(token) for token in x.tolist()]
    with open('output.txt', 'w') as f:
        f.write(''.join(tokens))

    for i in range(len(x)):
        color = "green" if y_mask[i] else "red"
        print_colored(f"('{tokens[i]}', {x[i]}, {y[i]})", color=color)

    console.print()

if __name__ == "__main__":
    fire.Fire(main)
