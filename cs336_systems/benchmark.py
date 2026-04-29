"""Perform end-to-end benchmarking of the forward pass, backward pass, and optimizer step."""

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import clip_gradient, cross_entropy
from cs336_basics.optimizer import AdamW
import timeit
import torch
import argparse
import numpy as np
import wandb
from einops import rearrange
import modal
import torch.cuda.nvtx as nvtx

from cs336_systems.modal_utils import DATA_PATH, VOLUME_MOUNTS, app, build_nsys_image

wandb_secret = modal.Secret.from_name("wandb")

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


@app.function(
    image=build_nsys_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    secrets=[wandb_secret],
    timeout=28800,
    max_containers=3,
)
def benchmark(args):
    torch.set_float32_matmul_precision("high")

    torch.cuda.memory._record_memory_history(max_entries=1000000)

    model = BasicsTransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta)
    model.to(device=args.device)
    # model = torch.compile(model)
    optimizer = AdamW(model.parameters(), args.lr, args.betas, args.eps, args.weight_decay)

    with nvtx.range("warmup steps"):
        for t in range(args.num_warmup_steps):
            inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
            targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward(inputs)
            loss = cross_entropy(rearrange(logits, "b c v -> (b c) v"), rearrange(targets, "b c -> (b c)"))
            if args.mode != "forward":
                loss.backward()
                clip_gradient(model.parameters(), args.max_grad_norm)
                if args.mode == "full":
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    steps_timer = [[] for _ in range(args.num_measurement_steps)]
    with nvtx.range("measurement"):
        for t in range(args.num_measurement_steps):
            inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
            targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
            # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            forward_start = timeit.default_timer()
            with nvtx.range("forward"):
                logits = model.forward(inputs)
                torch.cuda.synchronize()
            forward_time = timeit.default_timer() - forward_start
            steps_timer[t].append(forward_time)

            loss = cross_entropy(rearrange(logits, "b c v -> (b c) v"), rearrange(targets, "b c -> (b c)"))
            if args.mode != "forward":
                backward_start = timeit.default_timer()
                with nvtx.range("backward"):
                    loss.backward()
                    torch.cuda.synchronize()
                backward_time = timeit.default_timer() - backward_start
                steps_timer[t].append(backward_time)
                # clip_gradient(model.parameters(), args.max_grad_norm)
                if args.mode == "full":
                    optimizer_start = timeit.default_timer()
                    with nvtx.range("optimizer"):
                        optimizer.step()
                        torch.cuda.synchronize()
                    optimizer_time = timeit.default_timer() - optimizer_start
                    steps_timer[t].append(optimizer_time)
                optimizer.zero_grad(set_to_none=True)

    snapshot_path = f"/root/data/memory-noautocast-v2-{args.size}-cl{args.context_length}-{args.mode}.pickle"
    torch.cuda.memory._dump_snapshot(snapshot_path)
    torch.cuda.memory._record_memory_history(enabled=None)
    
    return steps_timer


@app.local_entrypoint()
def main(
    mode: str,
    size: str = "small",
    d_model: int = 512,
    num_heads: int = 16,
    num_layers: int = 4,
    d_ff: int = 1344,
    vocab_size: int = 10000,
    context_length: int = 512,
    rope_theta: float = 10000,
    lr: float = 1e-3,
    lr_min: float = 1e-4,
    betas: str = "0.9,0.999",
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    num_warmup_steps: int = 10,
    num_measurement_steps: int = 10,
    batch_size: int = 4,
    device: str = "cuda",
    max_grad_norm: float = 1.0,
):
    args = argparse.Namespace(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        vocab_size=vocab_size,
        context_length=context_length,
        rope_theta=rope_theta,
        lr=lr,
        lr_min=lr_min,
        betas=[float(x) for x in betas.split(",")],
        eps=eps,
        weight_decay=weight_decay,
        num_warmup_steps=num_warmup_steps,
        num_measurement_steps=num_measurement_steps,
        batch_size=batch_size,
        device=device,
        max_grad_norm=max_grad_norm,
        mode=mode,
        size=size,
    )

    size_config = MODEL_CONFIGS[args.size]
    for key, value in size_config.items():
        setattr(args, key, value)
    steps_timer = benchmark.remote(args)
    arr = np.array(steps_timer)

    if args.mode == "forward":
        names = ["forward"]
    elif args.mode == "backward":
        names = ["forward", "backward"]
    else:
        names = ["forward", "backward", "optimizer"]

    print()
    print(f"size={args.size}, mode={args.mode}")
    # print(f"d_model={args.d_model}, num_layers={args.num_layers}, num_heads={args.num_heads}, d_ff={args.d_ff}, batch_size={args.batch_size}, context_length={args.context_length}")

    for i, name in enumerate(names):
        mean = arr[:, i].mean()
        std = arr[:, i].std()
        print(f"{name}: {mean * 1000:.3f} ms +/- {std * 1000:.3f} ms")
