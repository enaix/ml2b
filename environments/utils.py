from pathlib import Path
import docker
from pydantic import BaseModel

class BuildArgs(BaseModel):
    """
    Docker image spec
    """
    dockerfile: str | Path
    platform: str
    rm: bool = True
    decode: bool = True
    path: str | Path = "."
    tag: str


def build_image(build_args: BuildArgs, client: docker.DockerClient | None = None, proxy_settings: dict[str, str] | None = None) -> None:
    """Build image from spec

    Args:
        build_args (BuildArgs): Image spec
        client (docker.DockerClient | None, optional): Docker client. Defaults to None.
        proxy_settings (dict[str, str] | None, optional): Proxy container build settings. Defaults to None.
    """
    if client is None:
        client = docker.from_env()
    for chunk in client.api.build(**build_args.model_dump(), buildargs=proxy_settings):
        if "stream" in chunk:
            for line in chunk["stream"].splitlines():
                print(line)
        elif "error" in chunk:
            print("ERROR:", chunk["error"])

