"""TOMLCloud benchmark: ResNet-18"""

import torch
from torchvision import models
from runner import run_benchmark, make_parser


def create_model():
    device = torch.device("cuda")
    model = models.resnet18(weights=None).to(device).eval()
    return model, device


def create_input(device, batch_size):
    return torch.randn(batch_size, 3, 224, 224, device=device)


if __name__ == "__main__":
    args = make_parser("resnet18").parse_args()
    run_benchmark(
        arch="resnet18",
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
