from pathlib import Path
import docker
from pydantic import BaseModel

class BuildArgs(BaseModel):
    """
    Base spec model for docker image
    """
    dockerfile: str | Path
    platform: str
    rm: bool = True
    decode: bool = True
    path: str | Path = "."
    tag: str


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

