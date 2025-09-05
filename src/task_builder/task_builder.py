from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template
from pydantic import BaseModel
from typing import Literal
BASE_TEMPLATE_PATH = Path(__file__).parent.resolve() / "bench_templates"

class TaskDescription(BaseModel):
    description: str
    domain: str
    metric: str
    datacard: str | None

class TaskContext(BaseModel):
    code_lang_extention: str
    code_lang: str
    competition_type_code: bool
    competition_type_file: bool
    code_template_variant: Literal["extended", "short"] = "extended"
    task_info: TaskDescription

class TaskBuilder:
    def __init__(self, templates_path: str = BASE_TEMPLATE_PATH):
        self.env = Environment(loader=FileSystemLoader(templates_path))
        self.base_template = self.env.get_template('base_instructions.j2')
    
    def render(self, context: TaskContext):
        rendered = self.base_template.render(**context.model_dump())
        return rendered
