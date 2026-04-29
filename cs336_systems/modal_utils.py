## put this in cs336_basics/modal_utils.py
from pathlib import Path, PurePosixPath

import modal

SUNET_ID = "alexskim"

if SUNET_ID == "":
    raise NotImplementedError(f"Please set the SUNET_ID in {__file__}")

(DATA_PATH := Path("data")).mkdir(exist_ok=True)

app = modal.App(f"basics-{SUNET_ID}")
user_volume = modal.Volume.from_name(f"basics-{SUNET_ID}", create_if_missing=True)


def build_image(*, include_tests: bool = False) -> modal.Image:
    image = modal.Image.debian_slim().apt_install("wget", "gzip").uv_sync()
    image = image.add_local_python_source("cs336_basics")
    image = image.add_local_python_source("cs336_systems")
    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")
    return image

def build_nsys_image(*, include_tests: bool = False) -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .run_commands(
            "apt-get update && apt-get install -y wget gnupg ca-certificates",
            "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update",
        )
        .apt_install("libcap2-bin", "libdw1", "cuda-nsight-systems-13-2")
        .uv_sync()
        .add_local_python_source("cs336_basics")
        .add_local_python_source("cs336_systems")
    )

    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")

    return image

VOLUME_MOUNTS: dict[str | PurePosixPath, modal.Volume | modal.CloudBucketMount] = {
    f"/root/{DATA_PATH}": user_volume,
}


def secrets(include_huggingface_secret: bool = False) -> list[modal.Secret]:
    secrets = [modal.Secret.from_dict({"SOME_ENV_VAR": "some-value"}), modal.Secret.from_name("my-secret")]
    return secrets
