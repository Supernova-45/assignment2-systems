import subprocess
import modal

from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_nsys_image


@app.function(
    image=build_nsys_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    timeout=32_400,
)
def run_nsys(
    size: str,
    mode: str,
    context_length: int = 512,
    batch_size: int = 4,
    num_warmup_steps: int = 5,
    num_measurement_steps: int = 1,
):
    output_base = f"/root/data/nsys-{size}-cl{context_length}-{mode}"

    cmd = [
        "nsys",
        "profile",
        "-o",
        output_base,
        "--force-overwrite=true",
        "--trace=cuda,nvtx,osrt",
        "--cuda-memory-usage=true",
        "--cudabacktrace=all",
        "--python-backtrace=cuda",
        "--gpu-metrics-devices=all",
        "--stats=true",
        "python",
        "-m",
        "cs336_systems.nsys_benchmark",
        "--size",
        size,
        "--mode",
        mode,
        "--context-length",
        str(context_length),
        "--batch-size",
        str(batch_size),
        "--num-warmup-steps",
        str(num_warmup_steps),
        "--num-measurement-steps",
        str(num_measurement_steps),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Saved to {output_base}.nsys-rep")


@app.local_entrypoint()
def main(
    size: str = "small",
    mode: str = "forward",
    context_length: int = 512,
    batch_size: int = 4,
    num_warmup_steps: int = 5,
    num_measurement_steps: int = 1,
):
    run_nsys.remote(
        size=size,
        mode=mode,
        context_length=context_length,
        batch_size=batch_size,
        num_warmup_steps=num_warmup_steps,
        num_measurement_steps=num_measurement_steps,
    )
