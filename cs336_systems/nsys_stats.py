import subprocess
import modal

from cs336_systems.modal_utils import VOLUME_MOUNTS, app, build_nsys_image


@app.function(
    image=build_nsys_image(),
    volumes=VOLUME_MOUNTS,
    gpu="B200",
    timeout=32_400,
)
def run_stats(report_name: str):
    report_path = f"/root/data/{report_name}"

    cmd = [
        "nsys",
        "stats",
        "--report",
        "cuda_gpu_kern_sum",
        report_path,
    ]

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(report_name: str):
    run_stats.remote(report_name)
