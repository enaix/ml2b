import os
from typing import Literal
from pydantic import BaseModel, FilePath, NewPath, DirectoryPath

class Config(BaseModel):
    model: str
    provider: Literal["openai", "vertex"]
    working_dir: NewPath | DirectoryPath
    env_file: FilePath
    data_dir: DirectoryPath
    description: FilePath
    temperature: float
    max_steps: int
    exp_name: str
    tool_timeout: float
