from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel
from typing import Literal, Any, Mapping
BASE_TEMPLATE_PATH = Path(__file__).parent.resolve() / "bench_templates"


def filter_dict_by_suffix(
    dct: dict[str, Any], 
    suffix: str,
    discard: bool = True,
    ignorecase: bool = True
) -> dict[str, Any]:
    suffix_lower = suffix.lower() if ignorecase else suffix
    if discard:
        return {key: val for key, val in dct.items() 
                if not key.lower().endswith(suffix_lower)}
    else:
        return {key: val for key, val in dct.items() 
                if key.lower().endswith(suffix_lower)}

def generate_google_args_doc(
    schema: Mapping[str, Any],
    *,
    indent: int = 4,
    container: str = "dict",
    sort_keys: bool = False,
) -> str:
    if container not in {"dict", "Mapping", "MutableMapping"}:
        raise ValueError("container must be 'dict', 'Mapping', or 'MutableMapping'")

    P = {
            "args": "Args:",
            "with_keys": {
                "dict": "dict with the following keys.",
                "Mapping": "Mapping with the following keys.",
                "MutableMapping": "MutableMapping with the following keys.",
            },
            "expected_keys": "Expected keys:",
        }

    IND = " " * indent

    def is_mapping(x: Any) -> bool:
        return isinstance(x, Mapping)

    def is_leaf(x: Any) -> bool:
        return is_mapping(x) and ("type" in x or "comment" in x)

    def union_str(types: list[str]) -> str:
        uniq = sorted(set(t for t in types if t))
        if not uniq:
            return "Any"
        if len(uniq) == 1:
            return uniq[0]
        return "Union[" + ", ".join(uniq) + "]"

    def fmt_container(inner: str) -> str:
        if container == "dict":
            return f"dict[str, {inner}]"
        return f"{container}[str, {inner}]"

    def collect_types(node: Any) -> list[str]:
        if is_leaf(node):
            return [str(node.get("type", "Any")) or "Any"]
        if is_mapping(node):
            collected: list[str] = []
            items = node.items()
            if sort_keys:
                items = sorted(items, key=lambda kv: kv[0])
            for _, v in items:
                if is_leaf(v):
                    collected.extend(collect_types(v))
                elif is_mapping(v):
                    inner = union_str(collect_types(v))
                    collected.append(fmt_container(inner))
                else:
                    collected.append("Any")
            return collected or ["Any"]
        return ["Any"]

    def mapping_type(node: Mapping[str, Any]) -> str:
        inner = union_str(collect_types(node))
        return fmt_container(inner)

    def emit_mapping_body(node: Mapping[str, Any], level: int, out: list[str]) -> None:
        items = node.items()
        if sort_keys:
            items = sorted(items, key=lambda kv: kv[0])

        for key, val in items:
            pad = IND * level
            if is_leaf(val):
                t = str(val.get("type", "Any")) or "Any"
                c = str(val.get("comment", "")).strip()
                out.append(f"{pad}{key} ({t}): {c}" if c else f"{pad}{key} ({t}):")
            elif is_mapping(val):
                t = mapping_type(val)
                out.append(f"{pad}{key} ({t}):")
                out.append(f"{pad}{IND}{P['expected_keys']}")
                emit_mapping_body(val, level + 2, out)
            else:
                out.append(f"{pad}{key} (Any):")

    lines: list[str] = [P["args"]]
    top_items = schema.items()
    if sort_keys:
        top_items = sorted(top_items, key=lambda kv: kv[0])

    for top_key, top_val in top_items:
        pad = IND
        if is_leaf(top_val):
            t = str(top_val.get("type", "Any")) or "Any"
            c = str(top_val.get("comment", "")).strip()
            lines.append(f"{pad}{top_key} ({t}): {c}" if c else f"{pad}{top_key} ({t}):")
        elif is_mapping(top_val):
            t = mapping_type(top_val)
            lines.append(f"{pad}{top_key} ({t}): {P['with_keys'][container]}")
            lines.append(f"{pad}{IND}{P['expected_keys']}")
            emit_mapping_body(top_val, 2, lines)
        else:
            lines.append(f"{pad}{top_key} (Any):")

    return "\n".join(lines)

def make_arg_types(schema: dict[str, Any], prefix: str = "") -> tuple[dict, list[str]]:
    typed_dict_defs = {}
    arg_annotations = {}

    for key, val in schema.items():
        if isinstance(val, dict):
            class_name = f"{key.capitalize()}Dict" if not prefix else f"{prefix}{key.capitalize()}Dict"
            fields = []
            for k2, v2 in val.items():
                fields.append(f"    {k2}: {v2}")
            typed_dict_code = f"class {class_name}(TypedDict):\n" + "\n".join(fields) + "\n"
            typed_dict_defs[class_name] = typed_dict_code
            arg_annotations[key] = class_name
        else:
            arg_annotations[key] = val
    return typed_dict_defs, arg_annotations

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
    full_schema: dict[str, Any]
    schema_dict: dict[str, Any]

class TaskBuilder:
    def __init__(self, templates_path: str = BASE_TEMPLATE_PATH):
        self.env = Environment(loader=FileSystemLoader(templates_path))
        self.base_template = self.env.get_template('base_instructions.j2')
    
    def render(self, context: TaskContext):
        task_context = context.model_dump()
        schema = task_context.get("full_schema")
        schema_dict = task_context.get("schema_dict")
        task_context["full_doc"] = generate_google_args_doc(schema)
        task_context["train_doc"] = generate_google_args_doc(filter_dict_by_suffix(schema, suffix="_val", discard=True))
        task_context["val_doc"] = generate_google_args_doc(filter_dict_by_suffix(schema, suffix="_val", discard=False))

        type_defs, arg_ann = make_arg_types(schema_dict)
        task_context["type_defs"] = type_defs
        task_context["full_spec"] = arg_ann
        task_context["train_spec"] = filter_dict_by_suffix(arg_ann, suffix="_val", discard=True)
        task_context["val_spec"] = filter_dict_by_suffix(arg_ann, suffix="_val", discard=False)

        rendered = self.base_template.render(**task_context)
        return rendered
