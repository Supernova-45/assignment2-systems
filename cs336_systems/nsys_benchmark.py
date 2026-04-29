"""Standalone workload for nsys profiling.

Run inside nsys with:
python -m cs336_systems.nsys_workload --size small --mode forward
"""

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import clip_gradient, cross_entropy
from cs336_basics.optimizer import AdamW
import timeit
import torch
import argparse
import numpy as np
from einops import rearrange
import torch.cuda.nvtx as nvtx


MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
    "10B": {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36,
    },
    "1024cl": {"d_model": 256, "d_ff": 1024, "num_layers": 4, "num_heads": 4, "context_length": 1024},
    "256cl": {"d_model": 256, "d_ff": 1024, "num_layers": 4, "num_heads": 4, "context_length": 256},
    "2048cl": {"d_model": 64, "d_ff": 256, "num_layers": 4, "num_heads": 4, "context_length": 2048},
}


def benchmark(args):
    torch.set_float32_matmul_precision("high")

    model = BasicsTransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
    )
    model.to(device=args.device)

    optimizer = AdamW(
        model.parameters(),
        args.lr,
        args.betas,
        args.eps,
        args.weight_decay,
    )

    with nvtx.range("warmup steps"):
        for t in range(args.num_warmup_steps):
            inputs = torch.randint(
                0,
                args.vocab_size,
                (args.batch_size, args.context_length),
                device=args.device,
            )
            targets = torch.randint(
                0,
                args.vocab_size,
                (args.batch_size, args.context_length),
                device=args.device,
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward(inputs)
                loss = cross_entropy(
                    rearrange(logits, "b c v -> (b c) v"),
                    rearrange(targets, "b c -> (b c)"),
                )

                if args.mode != "forward":
                    loss.backward()
                    clip_gradient(model.parameters(), args.max_grad_norm)

                    if args.mode == "full":
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()

    steps_timer = [[] for _ in range(args.num_measurement_steps)]

    with nvtx.range("measurement"):
        for t in range(args.num_measurement_steps):
            inputs = torch.randint(
                0,
                args.vocab_size,
                (args.batch_size, args.context_length),
                device=args.device,
            )
            targets = torch.randint(
                0,
                args.vocab_size,
                (args.batch_size, args.context_length),
                device=args.device,
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                forward_start = timeit.default_timer()

                with nvtx.range("forward"):
                    logits = model.forward(inputs)
                    torch.cuda.synchronize()

                forward_time = timeit.default_timer() - forward_start
                steps_timer[t].append(forward_time)

                loss = cross_entropy(
                    rearrange(logits, "b c v -> (b c) v"),
                    rearrange(targets, "b c -> (b c)"),
                )

                if args.mode != "forward":
                    backward_start = timeit.default_timer()

                    with nvtx.range("backward"):
                        loss.backward()
                        torch.cuda.synchronize()

                    backward_time = timeit.default_timer() - backward_start
                    steps_timer[t].append(backward_time)

                    if args.mode == "full":
                        optimizer_start = timeit.default_timer()

                        with nvtx.range("optimizer"):
                            optimizer.step()
                            torch.cuda.synchronize()

                        optimizer_time = timeit.default_timer() - optimizer_start
                        steps_timer[t].append(optimizer_time)

                    optimizer.zero_grad(set_to_none=True)

    return steps_timer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, choices=["forward", "backward", "full"])
    parser.add_argument("--size", type=str, default="small", choices=list(MODEL_CONFIGS.keys()))

    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1344)

    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--rope-theta", type=float, default=10000)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--num-warmup-steps", type=int, default=10)
    parser.add_argument("--num-measurement-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    args = parser.parse_args()

    size_config = MODEL_CONFIGS[args.size]
    for key, value in size_config.items():
        setattr(args, key, value)

    steps_timer = benchmark(args)
    arr = np.array(steps_timer)

    if args.mode == "forward":
        names = ["forward"]
    elif args.mode == "backward":
        names = ["forward", "backward"]
    else:
        names = ["forward", "backward", "optimizer"]

    print()
    print(f"size={args.size}, mode={args.mode}")

    for i, name in enumerate(names):
        mean = arr[:, i].mean()
        std = arr[:, i].std()
        print(f"{name}: {mean * 1000:.3f} ms +/- {std * 1000:.3f} ms")


if __name__ == "__main__":
    main()