from pathlib import Path
import docker
from pydantic import BaseModel

class BuildArgs(BaseModel):
    """
    Base spec model for docker image
    """
    dockerfile: str = "environments/runtime/Dockerfile"
    platform: str = "linux/amd64"
    rm: bool = True
    decode: bool = True
    path: str = "."
    tag: str = "runtime-env"


def build_image(build_args: BuildArgs, client: docker.DockerClient | None = None) -> None:
    """
    Build image from spec

    Args:
        build_args (BuildArgs): image spec
    """
    if client is None:
        client = docker.from_env()
    for chunk in client.api.build(**build_args.model_dump()):
        if "stream" in chunk:
            for line in chunk["stream"].splitlines():
                print(line)
        elif "error" in chunk:
            print("ERROR:", chunk["error"])


def build_agent(build_args: BuildArgs) -> None:
    """
    Build agent image from Dockerfile,
    if base runtime image not found previously build base runtime image

    Args:
        build_args (BuildArgs): image spec
    """
    client = docker.from_env()
    tag = "runtime-env"
    base_image = client.images.list(name=tag)
    build_base_runtime = len(base_image) == 0 and build_args.tag != tag
    if build_base_runtime:
        build_image(BuildArgs(), client)
    build_image(build_args, client)
