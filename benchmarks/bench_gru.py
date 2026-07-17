"""TOMLCloud benchmark: GRU (2-layer, 256 hidden)"""

import torch
import torch.nn as nn
from runner import run_benchmark, make_parser

INPUT_SIZE = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
SEQ_LENGTH = 100


def create_model():
    device = torch.device("cuda")
    model = nn.GRU(
        INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True
    ).to(device).eval()
    return model, device


def create_input(device, batch_size):
    return torch.randn(batch_size, SEQ_LENGTH, INPUT_SIZE, device=device)


if __name__ == "__main__":
    args = make_parser("gru").parse_args()
    run_benchmark(
        arch="gru",
        create_model_fn=create_model,
        create_input_fn=create_input,
        duration_s=args.duration,
        load_pct=args.load_pct,
        output=args.output,
        batch_size=args.batch_size,
        cooldown=not args.no_cooldown,
        cooldown_target_c=args.cooldown_target,
        window_stats_s=args.window_stats,
    )
