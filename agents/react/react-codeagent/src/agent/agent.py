from loguru import logger
import sys
import shutil
import os
from typing import Any
from pathlib import Path
import multiprocessing as mp
import json
from datetime import datetime
import time
from langchain_core.callbacks import UsageMetadataCallbackHandler

from langgraph.runtime import Runtime

from .router import get_llm_model
from ..config import Config
from .prompts import SYSTEM_PROMPT
import subprocess
import re


from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
)

from io import StringIO
from contextlib import contextmanager
from langchain.agents.middleware import (
    AgentMiddleware,
    before_model,
    AgentState,
    wrap_tool_call,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    after_agent,
)
from langchain.messages import ToolMessage, AIMessage
from langgraph.types import Command
from collections import deque
from langchain.tools.tool_node import ToolCallRequest
from typing import Callable
from langchain.tools import tool


def prepare_workflow(cfg: Config):
    base_path = cfg.working_dir / cfg.exp_name
    code_dir = base_path / "code"
    submission_dir = base_path / "submission"
    logs_dir = base_path / "logs"

    for dir in [code_dir, submission_dir, logs_dir]:
        shutil.rmtree(dir, ignore_errors=True)
        dir.mkdir(exist_ok=True, parents=True)

    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>",
    )
    logger.add(
        logs_dir / "agent.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function} | {message}",
        rotation="10 MB",
        retention="7 days",
    )

    return code_dir, submission_dir, logs_dir


@contextmanager
def capture_stdout():
    """Capture stdout for logging"""
    old_stdout = sys.stdout
    sys.stdout = buffer = StringIO()
    try:
        yield buffer
    finally:
        sys.stdout = old_stdout


@tool
def read_file(filepath: str, start_line: int = None, end_line: int = None) -> str:
    """Read content from a file. Useful for inspecting data files, scripts, or logs.

    Args:
        filepath (str): Path to the file to read
        start_line (int, optional): Starting line number (1-indexed). Defaults to None (read from beginning)
        end_line (int, optional): Ending line number (1-indexed, inclusive). Defaults to None (read to end)

    Returns:
        str: File content or error message
    """
    try:
        file_path = Path(filepath).resolve()

        if not file_path.exists():
            return f"Error: File '{filepath}' does not exist"

        if not file_path.is_file():
            return f"Error: '{filepath}' is not a file"

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            if start_line is not None or end_line is not None:
                lines = f.readlines()
                total_lines = len(lines)
                start = max(0, (start_line - 1) if start_line else 0)
                end = min(total_lines, end_line if end_line else total_lines)

                if start >= total_lines:
                    return f"Error: start_line {start_line} exceeds file length ({total_lines} lines)"

                content = "".join(lines[start:end])
                return f"Lines {start + 1}-{end} of {filepath} (total {total_lines} lines):\n{content}"
            else:
                content = f.read()
                num_lines = content.count("\n") + 1
                if len(content) > 10000:
                    preview = content[:10000]
                    return f"File {filepath} ({num_lines} lines, {len(content)} chars - showing first 10000 chars):\n{preview}\n\n[... truncated, use start_line/end_line to read specific parts]"
                return f"Content of {filepath} ({num_lines} lines):\n{content}"

    except UnicodeDecodeError:
        return f"Error: {filepath} appears to be a binary file"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def create_file(filepath: str, content: str) -> str:
    """Create a new file with given content. If file exists, it will be overwritten.
    Use this to create Python scripts, config files, or any text files.

    Args:
        filepath (str): Path to the file to create
        content (str): Content to write to the file

    Returns:
        str: Success message
    """
    try:
        file_path = Path(filepath).resolve()
        file_path.parent.mkdir(exist_ok=True, parents=True)

        exists = file_path.exists()

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        lines = content.count("\n") + 1
        action = "Overwritten" if exists else "Created"
        return f"{action} {filepath} ({lines} lines, {len(content)} chars)"

    except Exception as e:
        return f"Error creating file: {str(e)}"


_python_global_scope = {}


@tool
def run_python(filepath: str = None, arguments: str = "") -> str:
    """Run Python file.

    Args:
        filepath (str, optional): Path to Python file to execute (fresh process)
        arguments (str, optional): Command-line arguments

    Returns:
        str: Execution output
    """
    global _python_global_scope

    try:
        file_path = Path(filepath).resolve()

        if not file_path.exists():
            return f"Error: File '{filepath}' does not exist"

        result = subprocess.run(
            f"{sys.executable} {filepath} {arguments}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=600,
            cwd="/home",
            env=os.environ.copy(),
        )

        output = result.stdout + result.stderr
        if result.returncode == 0:
            return f"Script executed successfully:\n{output}"
        return f"Script failed (exit code {result.returncode}):\n{output}"

    except subprocess.TimeoutExpired:
        return "Execution timed out after 600 seconds"
    except Exception:
        import traceback

        return f"Error: {traceback.format_exc()}"


@tool
def list_directory(
    dirpath: str = "/home",
    max_items: int = 50,
    offset: int = 0,
    recursive: bool = False,
    max_depth: int = 2,
) -> str:
    """List files and subdirectories in a directory with pagination support.

    Args:
        dirpath (str): Path to directory to list. Defaults to /home
        max_items (int): Maximum number of items to return. Defaults to 50
        offset (int): Number of items to skip (for pagination). Defaults to 0
        recursive (bool): List subdirectories recursively. Defaults to False
        max_depth (int): Maximum recursion depth when recursive=True. Defaults to 2

    Returns:
        str: Directory listing or error message
    """
    try:
        dir_path = Path(dirpath).resolve()

        if not dir_path.exists():
            return f"Error: Directory '{dirpath}' does not exist"

        if not dir_path.is_dir():
            return f"Error: '{dirpath}' is not a directory"

        if recursive:
            items = []

            def scan_dir(path: Path, current_depth: int = 0, prefix: str = ""):
                if current_depth > max_depth:
                    return

                try:
                    sorted_items = sorted(
                        path.iterdir(), key=lambda x: (not x.is_dir(), x.name)
                    )
                    for item in sorted_items:
                        rel_path = item.relative_to(dir_path)
                        indent = "  " * current_depth

                        if item.is_dir():
                            items.append((f"{indent}📁 {rel_path}/", item, True))
                            scan_dir(item, current_depth + 1, prefix + "  ")
                        else:
                            size = item.stat().st_size
                            size_str = (
                                f"{size:,}B"
                                if size < 1024
                                else f"{size / 1024:.1f}KB"
                                if size < 1024 * 1024
                                else f"{size / (1024 * 1024):.1f}MB"
                            )
                            items.append(
                                (f"{indent} {rel_path} ({size_str})", item, False)
                            )
                except PermissionError:
                    items.append(
                        (f"{indent} {item.name}/ (permission denied)", item, True)
                    )

            scan_dir(dir_path)
            total_items = len(items)

            if total_items == 0:
                return f"Directory {dirpath} is empty (searched recursively, depth={max_depth})"

            # Apply pagination
            paginated_items = items[offset : offset + max_items]

            output = [f"Contents of {dirpath} (recursive, depth={max_depth}):"]
            output.append(
                f"Showing {offset + 1}-{offset + len(paginated_items)} of {total_items} items"
            )
            output.append("")

            for display_str, _, _ in paginated_items:
                output.append(display_str)

            if offset + max_items < total_items:
                remaining = total_items - (offset + max_items)
                output.append("")
                output.append(
                    f"... {remaining} more items. Use offset={offset + max_items} to see more."
                )

            return "\n".join(output)

        else:
            items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            total_items = len(items)

            if total_items == 0:
                return f"Directory {dirpath} is empty"

            paginated_items = items[offset : offset + max_items]

            output = [f"Contents of {dirpath}:"]
            output.append(
                f"Showing {offset + 1}-{offset + len(paginated_items)} of {total_items} items"
            )
            output.append("")

            dirs = [item for item in paginated_items if item.is_dir()]
            files = [item for item in paginated_items if item.is_file()]

            if dirs:
                output.append("Directories:")
                for item in dirs:
                    try:
                        num_children = len(list(item.iterdir()))
                        output.append(f" {item.name}/ ({num_children} items)")
                    except PermissionError:
                        output.append(f" {item.name}/ (permission denied)")

            if files:
                if dirs:
                    output.append("")
                output.append("Files:")
                for item in files:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f}MB"

                    ext = item.suffix if item.suffix else "no ext"
                    output.append(f"  {item.name} ({size_str}, {ext})")

            if offset + max_items < total_items:
                remaining = total_items - (offset + max_items)
                output.append("")
                output.append(
                    f"... {remaining} more items. Use offset={offset + max_items} to see more."
                )

            return "\n".join(output)

    except PermissionError:
        return f"Error: Permission denied to access '{dirpath}'"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def run_command(command: str) -> str:
    """Run a shell command and return the output

    Args:
        command (str): Shell command to execute

    Returns:
        str: Command output
    """
    if "<<" in command and ("'PY'" in command or '"PY"' in command):
        return (
            "Error: Heredoc syntax not supported in JSON.\n"
            "Instead:\n"
            "1. Use create_file() to save Python code\n"
            "2. Then run: python /home/yourscript.py"
        )
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            executable="/bin/bash",
            text=True,
            timeout=600,
            cwd="/home",
            env=os.environ.copy(),
        )
        if result.returncode == 0:
            return f"Command successed:\n{result.stdout}"
        return f"Command failed (exit code {result.returncode}):\n{result.stderr}\ncommand:\n{command}"
    except subprocess.TimeoutExpired:
        return "Command timed out after 600 seconds"
    except Exception as e:
        return f"Error running command: {e}"


class ToolCycleMiddleware(AgentMiddleware):
    def __init__(self, max_retries: int = 3, warning_retries: int = 2):
        super().__init__()
        self.max_retries = max_retries
        self.warning_retries = warning_retries
        self.tool_seq = deque(maxlen=max_retries)

    def _gen_tool_warn(self, num_retries: int, tool_name: str, tool_arg: str) -> str:
        if num_retries == self.max_retries:
            return (
                f"CRITICAL: You appear to be stuck in a loop with '{tool_name}'!\n"
                f"This tool has been called repeatedly without progress.\n"
                f"REQUIRED ACTIONS:\n"
                f"Try a completely different approach\n"
                f"Consider if the task needs to be broken down differently\n"
                f"Review what you've tried and why it hasn't worked\n"
            )
        if num_retries == self.warning_retries:
            return (
                f"Warning: '{tool_name}' has been called multiple times with similar results. "
                f"Consider:\n"
                f"Using a different tool\n"
                f"Changing your approach\n"
                f"Re-evaluating the problem"
            )
        return ""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ):
        tool_name = request.tool_call["name"]
        tool_args = str(sorted(request.tool_call["args"].items()))
        tool_signature = f"{tool_name}::{tool_args}"

        self.tool_seq.append(tool_signature)
        recent = list(self.tool_seq)
        last_call = recent[0]
        i = -1
        for tool_call in recent:
            if tool_call == last_call:
                i += 1
        tool_warn = self._gen_tool_warn(i, tool_name, tool_args)

        try:
            result = handler(request)
            if tool_warn and isinstance(result, ToolMessage):
                result.content = f"{result.content}\n\n{tool_warn}"
            return result
        except Exception as e:
            return ToolMessage(
                content=f"Tool '{tool_name}' failed: {e}",
                tool_call_id=request.tool_call["id"],
            )


def run_with_timeout(func, args, kwargs, timeout):
    """Run function in separate process with timeout"""

    def wrapper(queue, func, args, kwargs):
        try:
            result = func(*args, **kwargs)
            queue.put(("success", result))
        except Exception as e:
            queue.put(("error", e))

    queue = mp.Queue()
    process = mp.Process(target=wrapper, args=(queue, func, args, kwargs))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Function timed out after {timeout} seconds")

    if not queue.empty():
        status, result = queue.get()
        if status == "error":
            raise result
        return result

    raise RuntimeError("Process terminated without result")


@wrap_tool_call
def tools_time_limit(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    tool_timeout = request.runtime.context["tool_timeout"]
    start_time = time.monotonic()

    try:
        tool_name = request.tool_call["name"]

        if tool_name == "manage_todo":
            return handler(request)

        result = run_with_timeout(handler, (request,), {}, tool_timeout)
        return result

    except TimeoutError:
        return ToolMessage(
            content=f"Error: Tool execution timed out after {tool_timeout} seconds",
            tool_call_id=request.tool_call["id"],
        )
    except Exception as e:
        elapsed = time.monotonic() - start_time
        return ToolMessage(
            content=f"Tool '{request.tool_call['name']}' failed after {elapsed:.2f}s: {e}",
            tool_call_id=request.tool_call["id"],
        )


def format_time(time_in_sec: int):
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {int(time_in_sec % 60)}secs"


@before_model(can_jump_to=["end"])
def agent_limitations(state: AgentState, runtime: Runtime) -> dict[str | Any] | None:
    steps_remaining = runtime.context.get("steps_remaining")
    time_remaining = runtime.context.get("time_remaining")
    step_time = runtime.context.get("step_time")
    if steps_remaining is None:
        runtime.context["steps_remaining"] = 100
        steps_remaining = 100
    if time_remaining is None:
        runtime.context["time_remaining"] = 600
        time_remaining = 600
    if step_time is None:
        runtime.context["step_time"] = 0
        step_time = 0

    runtime.context["steps_remaining"] = steps_remaining - 1
    if steps_remaining == 0:
        return {
            "messages": [AIMessage("Conversation limit reached.")],
            "jump_to": "end",
        }
    if steps_remaining == 1:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "This is your last answer, make sure submission code is available.",
                }
            ]
        }

    time_remaining = time_remaining - (time.monotonic() - step_time)
    reminder_msg = {
        "role": "system",
        "content": f"<TOTAL TIME REMAINING: {format_time(time_remaining)}>\n<TOTAL STEPS REMAINING: {steps_remaining}>",
    }
    runtime.context["step_time"] = time.monotonic()
    runtime.context["time_remaining"] = time_remaining
    return {"messages": [reminder_msg]}


@wrap_model_call
def handle_parse_errors(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for i in range(5):
        try:
            return handler(request)
        except Exception as e:
            if i == 4:
                raise
            request.messages = request.messages + [
                {"role": "user", "content": f"Error: {e}"}
            ]


_todo_state = {"tasks": [], "completed": [], "created_at": None}


@tool
def manage_todo(action: str, task: str = None, task_id: int = None) -> str:
    """Manage your TODO list for the ML task.

    REQUIRED FIRST STEP: Call with action='create' to initialize your plan!

    Args:
        action (str): Action to perform. Options:
            - 'create': Initialize TODO list with tasks (provide task as multiline string)
            - 'add': Add a single task
            - 'complete': Mark task as done (provide task_id)
            - 'view': View current TODO list
            - 'progress': Show progress summary
        task (str): Task description (for 'create' and 'add' actions).
                   For 'create', use multiline string like:
                   "1. Read data\n2. Build baseline\n3. Improve model"
        task_id (int): Task ID to complete (for 'complete' action)

    Returns:
        str: Status message or TODO list view
    """
    global _todo_state

    if action == "create":
        if not task:
            return "Error: 'create' requires 'task' parameter with your plan"

        # Parse tasks from string
        lines = [line.strip() for line in task.split("\n") if line.strip()]
        tasks = []
        for line in lines:
            # Remove numbering like "1.", "- ", etc
            cleaned = re.sub(r"^[\d\.\-\*\[\]\s]+", "", line)
            if cleaned:
                tasks.append(cleaned)

        _todo_state = {
            "tasks": tasks,
            "completed": [False] * len(tasks),
            "created_at": datetime.now().isoformat(),
        }

        return (
            f"✓ TODO list created with {len(tasks)} tasks:\n\n"
            + "\n".join(f"{i + 1}. [ ] {t}" for i, t in enumerate(tasks))
            + "\n\nNow start working on task 1!"
        )

    if action == "add":
        if not task:
            return "Error: 'add' requires 'task' parameter"

        if not _todo_state["created_at"]:
            return "Error: TODO list not initialized. Use action='create' first!"

        _todo_state["tasks"].append(task)
        _todo_state["completed"].append(False)
        task_num = len(_todo_state["tasks"])

        return f"✓ Added task #{task_num}: {task}"

    if action == "complete":
        if task_id is None:
            return "Error: 'complete' requires 'task_id' parameter"

        if not _todo_state["created_at"]:
            return "Error: TODO list not initialized"

        idx = task_id - 1
        if idx < 0 or idx >= len(_todo_state["tasks"]):
            return f"Error: Invalid task_id {task_id}. Valid range: 1-{len(_todo_state['tasks'])}"

        if _todo_state["completed"][idx]:
            return f"Task #{task_id} was already completed"

        _todo_state["completed"][idx] = True
        completed_count = sum(_todo_state["completed"])
        total_count = len(_todo_state["tasks"])

        return (
            f"✓ Completed task #{task_id}: {_todo_state['tasks'][idx]}\n"
            f"Progress: {completed_count}/{total_count} tasks done "
            f"({100 * completed_count // total_count}%)"
        )

    if action == "view":
        if not _todo_state["created_at"]:
            return "No TODO list exists. Create one with action='create'!"

        output = ["Current TODO List:", ""]
        for i, (task, done) in enumerate(
            zip(_todo_state["tasks"], _todo_state["completed"])
        ):
            status = "✓" if done else " "
            output.append(f"{i + 1}. [{status}] {task}")

        completed = sum(_todo_state["completed"])
        total = len(_todo_state["tasks"])
        output.append("")
        output.append(
            f"Progress: {completed}/{total} completed ({100 * completed // total}%)"
        )

        return "\n".join(output)

    if action == "progress":
        if not _todo_state["created_at"]:
            return "No TODO list exists yet"

        completed = sum(_todo_state["completed"])
        total = len(_todo_state["tasks"])
        pct = 100 * completed // total if total > 0 else 0

        next_task = None
        for i, (task, done) in enumerate(
            zip(_todo_state["tasks"], _todo_state["completed"])
        ):
            if not done:
                next_task = (i + 1, task)
                break

        result = [
            f"Progress: {completed}/{total} tasks ({pct}%)",
            f"Completed: {completed}",
            f"Remaining: {total - completed}",
        ]

        if next_task:
            result.append(f"\nNext task: #{next_task[0]} - {next_task[1]}")
        else:
            result.append("\n All tasks completed!")

        return "\n".join(result)

    return f"Error: Unknown action '{action}'. Valid: create, add, complete, view, progress"


@after_agent(can_jump_to="model")
def check_todo(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    global _todo_state
    steps_remaining = runtime.context.get("steps_remaining")
    if len(_todo_state["completed"]) == 0 and steps_remaining > 0:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "Todo not initialized, initialize and complete it before finish",
                }
            ],
            "jump_to": "model",
        }
    if not all(_todo_state["completed"]) and steps_remaining > 0:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": f"Not all todo tasks are completed, complete tasks before finish:\n{manage_todo.run({'action': 'view'})}",
                }
            ],
            "jump_to": "model",
        }


def run_react(cfg: Config):
    """Runs ReAct agent for ml tasks"""
    _, _, logs_dir = prepare_workflow(cfg)
    with open(cfg.description, "r", encoding="utf-8") as td:
        task_description = td.read()
    llm = get_llm_model(cfg.provider, cfg.model, cfg.temperature)
    agent = create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            create_file,
            run_command,
            read_file,
            run_python,
            list_directory,
            manage_todo,
        ],
        middleware=[
            agent_limitations,
            ToolCycleMiddleware(),
            tools_time_limit,
            handle_parse_errors,
            SummarizationMiddleware(
                model=llm, trigger=("tokens", 20000), keep=("messages", 15)
            ),
            check_todo,
        ],
    )

    callback = UsageMetadataCallbackHandler()
    for step in agent.stream(
        {"messages": [{"role": "user", "content": task_description}]},
        stream_mode="values",
        context={
            "steps_remaining": cfg.max_steps,
            "time_remaining": cfg.time_limit,
            "step_time": time.monotonic(),
            "tool_timeout": cfg.tool_timeout,
            "max_steps": cfg.max_steps,
        },
        config={"callbacks": [callback]},
    ):
        with capture_stdout() as output:
            step["messages"][-1].pretty_print()
        logger.info(output.getvalue())
    with open(logs_dir / "usage_metadata.json", "w", encoding="utf-8") as f:
        json.dump(callback.usage_metadata, f, indent=2)
