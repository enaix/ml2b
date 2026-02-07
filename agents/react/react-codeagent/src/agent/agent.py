import asyncio
from typing import Annotated, Literal
from pydantic import BaseModel, DirectoryPath
from loguru import logger
import sys
import shutil
from typing import Any
from pathlib import Path

from langchain_core.messages import AnyMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from .prompts import build_system_prompt
from .router import get_llm_model
from ..config import Config
from .tools import get_tools
import subprocess
import re

class ContextSchema(BaseModel):
    chat_model: Any
    tools_by_name: dict[str, Any]
    task_description: str
    tool_timeout: float
    code_dir: DirectoryPath
    data_dir: DirectoryPath
    submission_dir: DirectoryPath


class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    remaining_steps: int
    iterations: int = 0
    submission_ready: bool = False
    best_score: float = 0.0 
    last_validation_iter: int = 0 


async def execute_tool(tool_call: dict[str, Any], tools: dict[str, Any], timeout: float = 3600):
    try:
        tool = tools[tool_call["name"]]
        if not tool:
            return tool_call["id"], f"Error: Tool '{tool_call['name']}' not found"
        result = await asyncio.wait_for(
            asyncio.to_thread(tool.invoke, tool_call["args"]), timeout=timeout
        )
        
        result_str = str(result)
        if len(result_str) > 4000:
            result_str = (
                result_str[:2000] + 
                f"\n... [output truncated, {len(result_str)} total chars] ...\n" + 
                result_str[-2000:]
            )
            result = result_str
        
        return tool_call["id"], result
    
    except TimeoutError:
        return tool_call["id"], f"Error: Tool execution timed out after {timeout}s"
    except Exception as e:
        logger.exception(f"Tool execution error for {tool_call['name']}")
        return tool_call["id"], f"Error executing tool: {str(e)}"


async def execute_tools(state: AgentState, runtime: Runtime[ContextSchema]) -> dict:
    tasks = []
    tool_timeout = runtime.context.tool_timeout
    tools = runtime.context.tools_by_name
    
    for tool_call in state.messages[-1].tool_calls:
        logger.info("Tool call: {}:{} with args: {}", tool_call["name"], tool_call["id"], tool_call["args"])
        tasks.append(execute_tool(tool_call, tools, tool_timeout))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    tools_results = []
    
    for (id, res) in results:
        logger.info("Tool call: {} result: {}", id, res[:500] if len(str(res)) > 500 else res)
        tools_results.append(ToolMessage(content=res, tool_call_id=id))
    
    return {"messages": tools_results, "iterations": state.iterations + 1}


async def call_model(state: AgentState, runtime: Runtime[ContextSchema]) -> dict:
    tools = runtime.context.tools_by_name.values()
    chat_model_with_tools = runtime.context.chat_model.bind_tools(tools)
    response = await chat_model_with_tools.ainvoke(state.messages)
    return {"messages": [response]}


async def cleanup_messages(state: AgentState) -> dict:
    messages = state.messages
    cleaned = []
    
    for msg in messages:
        if msg.type in ["system", "ai"]:
            cleaned.append(msg)
            continue
        
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            if len(content) > 3000:
                truncated = (
                    content[:1500] + 
                    f"\n... [truncated {len(content)-3000} chars] ...\n" + 
                    content[-1500:]
                )
                cleaned.append(ToolMessage(content=truncated, tool_call_id=msg.tool_call_id))
            else:
                cleaned.append(msg)
        else:
            cleaned.append(msg)
    
    if len(cleaned) < len(messages):
        logger.info(f"Cleaned messages: {len(messages)} -> {len(cleaned)}")
        return {"messages": cleaned}
    
    return {}


async def summarize_history(state: AgentState, runtime: Runtime[ContextSchema]) -> dict:
    messages = state.messages
    
    SUMMARIZE_THRESHOLD = 25
    KEEP_RECENT = 15
    
    total_messages = len(messages)
    
    if total_messages < SUMMARIZE_THRESHOLD:
        return {}
    
    # Системные сообщения
    system_indices = [i for i, msg in enumerate(messages) if msg.type == "system"]
    first_non_system_idx = max(system_indices) + 1 if system_indices else 0
    
    system_messages = messages[:first_non_system_idx]
    non_system_messages = messages[first_non_system_idx:]
    
    if len(non_system_messages) < KEEP_RECENT:
        return {}
    
    recent_messages = non_system_messages[-KEEP_RECENT:]
    messages_to_summarize = non_system_messages[:-KEEP_RECENT]
    
    # Улучшенный промпт для резюме
    summary_content = []
    for msg in messages_to_summarize:
        if hasattr(msg, 'content'):
            content_str = str(msg.content)[:500]
            summary_content.append(f"{msg.type}: {content_str}")
    
    summary_prompt = f"""Summarize this ML agent's work, preserving KEY FACTS:

1. **Best score achieved**: What was the highest metric score?
2. **Key decisions**: What models/approaches were tried?
3. **Current state**: What files exist in submission/?
4. **Critical issues**: Any errors or blockers?

History:
{chr(10).join(summary_content)}

Current best score from state: {state.best_score:.6f}

Write a concise summary (max 300 tokens) that preserves the best score and key progress."""
    
    chat_model = runtime.context.chat_model
    summary_response = await chat_model.ainvoke([
        {"role": "user", "content": summary_prompt}
    ])
    
    summary_text = (
        f"[HISTORY SUMMARY]\n"
        f"Best score so far: {state.best_score:.6f}\n\n"
        f"{summary_response.content}"
    )
    
    summary_message = AIMessage(content=summary_text)
    
    new_messages = [*system_messages, summary_message, *recent_messages]
    
    logger.info(
        f"Summarized {len(messages_to_summarize)} messages. "
        f"New: {len(new_messages)} (kept best_score: {state.best_score:.6f})"
    )
    
    return {"messages": new_messages}


async def progress_reminder(state: AgentState) -> dict:
    """
    Напоминает о прогрессе И о проверке метрики.
    """
    iterations = state.iterations
    remaining = state.remaining_steps
    
    if iterations % 5 == 0 or remaining <= 5:
        if remaining <= 2:
            msg = (
                f"🚨 CRITICAL: {remaining} steps left!\n"
                f"Finalize submission NOW."
            )
        elif remaining <= 5:
            msg = (
                f"⚠️ {remaining} steps remaining.\n"
                f"Make sure you:\n"
                f"1. Created submission/submission.py\n"
                f"2. Called validate_submission() to check score\n"
                f"3. Ready to finish()"
            )
        elif iterations % 10 == 0:
            msg = (
                f"Progress check ({iterations} iterations, {remaining} steps):\n"
                f"- Have you measured your solution's score?\n"
                f"- Use validate_submission() to check quality"
            )
        else:
            msg = f"Progress: {iterations} iterations, {remaining} steps remaining"
        
        return {"messages": [{"role": "user", "content": msg}]}
    
    return {}

def extract_score(output: str) -> float | None:
    """Extract metric score from submission output.
    
    Tries multiple common patterns to find the score.
    Adapt this function to match your benchmark's output format.
    
    Args:
        output: stdout from running submission.py
        
    Returns:
        Score as float, or None if not found
    """
    
    # Паттерны для поиска метрики (добавьте свои форматы!)
    patterns = [
        r'Score:\s*([\d.]+)',           # "Score: 0.85"
        r'Metric:\s*([\d.]+)',          # "Metric: 0.85"
        r'Accuracy:\s*([\d.]+)',        # "Accuracy: 0.85"
        r'F1:\s*([\d.]+)',              # "F1: 0.85"
        r'AUC:\s*([\d.]+)',             # "AUC: 0.85"
        r'RMSE:\s*([\d.]+)',            # "RMSE: 0.15"
        r'MAE:\s*([\d.]+)',             # "MAE: 0.15"
        r'R2:\s*([\d.]+)',              # "R2: 0.85"
        r'(?:^|\n)(\d+\.\d+)(?:\n|$)',  # Просто число на отдельной строке
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                # Базовая валидация (метрики обычно 0-1 или 0-100)
                logger.info(f"Extracted score: {score} using pattern '{pattern}'")
                return score
            except ValueError:
                continue
    
    logger.warning(f"Could not extract score from output:\n{output[:500]}")
    return None

async def submission_validator(state: AgentState, runtime: Runtime[ContextSchema]) -> dict:
    submission_dir = runtime.context.submission_dir
    submission_file = submission_dir / "submission.py"
    
    if not submission_file.exists():
        return {
            "messages": [{
                "role": "user", 
                "content": (
                    "❌ No submission.py\n"
                    "Create it using write_file('submission/submission.py', ...)"
                )
            }],
            "submission_ready": False
        }
    
    # Проверяем структуру кода
    try:
        with open(submission_file, 'r') as f:
            content = f.read()
        
        required_functions = ['train', 'prepare_val', 'predict', 'run']
        missing = [f for f in required_functions if f"def {f}(" not in content]
        
        if missing:
            return {
                "messages": [{
                    "role": "user",
                    "content": f"❌ Missing functions: {missing}\nFix your submission."
                }],
                "submission_ready": False
            }
        
        # 🔥 АВТОМАТИЧЕСКИ ЗАПУСКАЕМ ПРОВЕРКУ МЕТРИКИ
        logger.info("Running automatic score validation...")
        result = subprocess.run(
            ["uv", "run", "python", str(submission_file)],
            capture_output=True,
            text=True,
            timeout=runtime.context.tool_timeout,
            env={"DATA_DIR": str(runtime.context.data_dir.absolute())}
        )
        
        if result.returncode != 0:
            return {
                "messages": [{
                    "role": "user",
                    "content": (
                        f"❌ Submission execution failed:\n"
                        f"{result.stderr[:1500]}\n\n"
                        f"Fix the errors and try again."
                    )
                }],
                "submission_ready": False
            }
        
        # Извлекаем метрику
        output = result.stdout
        score = extract_score(output)
        
        if score is None:
            return {
                "messages": [{
                    "role": "user",
                    "content": (
                        f"⚠️ Submission ran but no score found.\n"
                        f"Output:\n{output[:1000]}\n\n"
                        f"Check if submission.py outputs the metric correctly."
                    )
                }],
                "submission_ready": False
            }
        
        # Сохраняем лучший результат
        best_score = state.get("best_score", float("inf"))
        if score > best_score:
            logger.info(f"New best score: {score} (previous: {best_score})")
            return {
                "messages": [{
                    "role": "user",
                    "content": (
                        f"✅ NEW BEST SCORE: {score:.4f}! 🎉\n"
                        f"Previous best: {best_score:.4f}\n\n"
                        f"Submission is valid. You can:\n"
                        f"- Continue improving ({state.remaining_steps} steps left)\n"
                        f"- Call finish() if satisfied"
                    )
                }],
                "submission_ready": True,
                "best_score": score
            }
        else:
            return {
                "messages": [{
                    "role": "user",
                    "content": (
                        f"✓ Score: {score:.4f}\n"
                        f"Best score: {best_score:.4f}\n\n"
                        f"This version is worse. Revert or continue iterating."
                    )
                }],
                "submission_ready": True,
                "best_score": best_score
            }
        
    except subprocess.TimeoutExpired:
        return {
            "messages": [{"role": "user", "content": "❌ Submission timeout (>300s)"}],
            "submission_ready": False
        }
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {
            "messages": [{"role": "user", "content": f"❌ Validation error: {str(e)}"}],
            "submission_ready": False
        }


async def step_decrementer(state: AgentState) -> dict:
    last_message = state.messages[-1]
    
    cheap_tools = {'list_files', 'read_file'}
    
    if isinstance(last_message, ToolMessage):
        for msg in reversed(state.messages[:-1]):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc['id'] == last_message.tool_call_id:
                        if tc['name'] in cheap_tools:
                            logger.debug(f"Cheap tool {tc['name']}, not decrementing steps")
                            return {}
                        break
    
    new_steps = state.remaining_steps - 1
    logger.info(f"Steps: {state.remaining_steps} -> {new_steps}")
    return {"remaining_steps": new_steps}


async def react_router(state: AgentState) -> Literal["__end__", "tools", "validate", "progress"]:
    if state.remaining_steps <= 3:
        logger.info("Low steps - forcing validation")
        return "validate"
    
    if state.remaining_steps <= 0:
        if state.submission_ready and state.get("best_score", 0) > 0:
            logger.info("Steps exhausted, valid submission exists")
            return "__end__"
        else:
            logger.warning("No valid submission yet, forcing validation")
            return "validate"
    
    last_message = state.messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # Валидация после каждых 5 итераций (не 10)
    if state.iterations > 0 and state.iterations % 5 == 0:
        logger.info("Periodic validation check")
        return "validate"
    
    return "progress"

async def tools_router(state: AgentState) -> Literal["validate", "cleanup"]:
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            for ai_msg in reversed(state.messages):
                if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
                    for tc in ai_msg.tool_calls:
                        if tc['id'] == msg.tool_call_id and tc['name'] == 'finish':
                            logger.info("Finish tool detected, routing to validation")
                            return "validate"
                    break
            break

        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            break
    
    return "cleanup"


def create_graph():
    builder = StateGraph(AgentState, context_schema=ContextSchema)
    
    builder.add_node(execute_tools)
    builder.add_node(call_model)
    builder.add_node(submission_validator)
    builder.add_node(progress_reminder)
    builder.add_node(step_decrementer)
    builder.add_node(cleanup_messages)
    builder.add_node(summarize_history)

    builder.add_edge("__start__", "call_model")
    
    builder.add_conditional_edges(
        "call_model", 
        react_router, 
        {
            "tools": "execute_tools",
            "validate": "submission_validator", 
            "progress": "progress_reminder",
            "__end__": "__end__"
        }
    )
    builder.add_conditional_edges(
        "execute_tools",
        tools_router,
        {
            "validate": "submission_validator",
            "cleanup": "cleanup_messages"
        }
    )
    builder.add_edge("cleanup_messages", "step_decrementer")
    builder.add_edge("step_decrementer", "summarize_history")
    builder.add_edge("summarize_history", "call_model")
    
    builder.add_edge("submission_validator", "call_model")
    builder.add_edge("progress_reminder", "call_model")
    
    return builder.compile()


async def _run_agent(context: ContextSchema, cfg: Config):
    graph = create_graph()
    logger.info("Agentic runtime graph:\n{}", graph.get_graph().draw_mermaid())
    
    messages = [
        {"role": "system", "content": build_system_prompt(context.data_dir.absolute(), context.code_dir.absolute(), context.submission_dir.absolute())},
        {
            "role": "system", 
            "content": (
                f"TASK INSTRUCTIONS:\n{cfg.description}\n\n"
                f"TASK INSTRUCTIONS IS NOST IMPORTANT OVER ANOTHER INSTRUCTIONS"
                f"IMPORTANT: DATA PATHS\n"
                f"The data directory absolute path is: {cfg.data_dir.absolute()}\n"
                f"Working directories:\n"
                f"- Data: {cfg.data_dir.absolute()} (READ-ONLY)\n"
                f"- Code: {context.code_dir.absolute()} (your workspace)\n"
                f"- Submission: {context.submission_dir.absolute()} (final files)\n\n"
                f"YOUR WORKFLOW:..."
            )
        }
    ]
    
    result = await graph.ainvoke(
        {
            "messages": messages,
            "remaining_steps": cfg.max_steps,
            "iterations": 0,
            "submission_ready": False
        }, 
        context=context, 
        config={"recursion_limit": sys.maxsize}
    )
    
    logger.info("Agent finished.")
    logger.info("Submission ready: {}", result.get("submission_ready", False))
    logger.info("Total iterations: {}", result.get("iterations", 0))
    logger.info("Total messages: {}", len(result.get("messages", [])))
    
    submission_files = list(context.submission_dir.glob("*.py"))
    if submission_files:
        logger.info("Submission file: {}", submission_files[0].name)
    else:
        logger.error("No submission file created!")


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


def run_agent(cfg: Config):
    code_dir, submission_dir = prepare_workflow(cfg)
    
    with open(cfg.description, "r", encoding="utf-8") as td:
        task_description = td.read()
    
    llm = get_llm_model(cfg.provider, cfg.model, cfg.temperature)
    tools_by_name = get_tools(
        code_dir=code_dir, 
        data_dir=cfg.data_dir,
        submission_dir=submission_dir
    )
    
    context = ContextSchema(
        chat_model=llm, 
        tools_by_name=tools_by_name,
        task_description=task_description,
        tool_timeout=cfg.tool_timeout,
        code_dir=code_dir,
        data_dir=cfg.data_dir,
        submission_dir=submission_dir
    )
    
    logger.info("Agent configuration: {}", context.model_dump_json(indent=2, exclude=["tools_by_name", "chat_model"]))
    logger.info("Agent tools: {}", list(tools_by_name.keys()))
    
    asyncio.run(_run_agent(context, cfg))




from langchain.agents import create_agent
from langchain.agents.middleware import (SummarizationMiddleware,
                                         TodoListMiddleware,
                                         LLMToolSelectorMiddleware,
                                         ToolRetryMiddleware
                                        )

from io import StringIO
import sys
from contextlib import contextmanager
from langchain.agents.middleware import AgentMiddleware, before_model, AgentState, wrap_tool_call, wrap_model_call, ModelRequest, ModelResponse, after_agent, hook_config
from langchain.messages import ToolMessage
from langgraph.types import Command
from collections import defaultdict, deque
from langchain.tools.tool_node import ToolCallRequest
from langchain.messages import AIMessage
from typing import Callable
from langchain.tools import tool
import os
import re


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
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            if start_line is not None or end_line is not None:
                lines = f.readlines()
                total_lines = len(lines)
                start = max(0, (start_line - 1) if start_line else 0)
                end = min(total_lines, end_line if end_line else total_lines)
                
                if start >= total_lines:
                    return f"Error: start_line {start_line} exceeds file length ({total_lines} lines)"
                
                content = "".join(lines[start:end])
                return f"Lines {start+1}-{end} of {filepath} (total {total_lines} lines):\n{content}"
            else:
                content = f.read()
                num_lines = content.count('\n') + 1
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
        
        lines = content.count('\n') + 1
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
            env=os.environ.copy()
        )
        
        output = result.stdout + result.stderr
        if result.returncode == 0:
            return f"Script executed successfully:\n{output}"
        return f"Script failed (exit code {result.returncode}):\n{output}"
        
    except subprocess.TimeoutExpired:
        return f"Execution timed out after 600 seconds"
    except Exception as e:
        import traceback
        return f"Error: {traceback.format_exc()}"


@tool
def list_directory(dirpath: str = "/home", max_items: int = 50, offset: int = 0, recursive: bool = False, max_depth: int = 2) -> str:
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
                    sorted_items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                    for item in sorted_items:
                        rel_path = item.relative_to(dir_path)
                        indent = "  " * current_depth
                        
                        if item.is_dir():
                            items.append((f"{indent}📁 {rel_path}/", item, True))
                            scan_dir(item, current_depth + 1, prefix + "  ")
                        else:
                            size = item.stat().st_size
                            size_str = f"{size:,}B" if size < 1024 else f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/(1024*1024):.1f}MB"
                            items.append((f"{indent}📄 {rel_path} ({size_str})", item, False))
                except PermissionError:
                    items.append((f"{indent} {item.name}/ (permission denied)", item, True))
            
            scan_dir(dir_path)
            total_items = len(items)
            
            if total_items == 0:
                return f"Directory {dirpath} is empty (searched recursively, depth={max_depth})"
            
            # Apply pagination
            paginated_items = items[offset:offset + max_items]
            
            output = [f"Contents of {dirpath} (recursive, depth={max_depth}):"]
            output.append(f"Showing {offset + 1}-{offset + len(paginated_items)} of {total_items} items")
            output.append("")
            
            for display_str, _, _ in paginated_items:
                output.append(display_str)
            
            if offset + max_items < total_items:
                remaining = total_items - (offset + max_items)
                output.append("")
                output.append(f"... {remaining} more items. Use offset={offset + max_items} to see more.")
            
            return "\n".join(output)
        
        else:
            items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            total_items = len(items)
            
            if total_items == 0:
                return f"Directory {dirpath} is empty"
            
            paginated_items = items[offset:offset + max_items]
            
            output = [f"Contents of {dirpath}:"]
            output.append(f"Showing {offset + 1}-{offset + len(paginated_items)} of {total_items} items")
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
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size/(1024*1024):.1f}MB"
                    
                    ext = item.suffix if item.suffix else "no ext"
                    output.append(f"  {item.name} ({size_str}, {ext})")
            
            if offset + max_items < total_items:
                remaining = total_items - (offset + max_items)
                output.append("")
                output.append(f"... {remaining} more items. Use offset={offset + max_items} to see more.")
            
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
    if '<<' in command and ("'PY'" in command or '"PY"' in command):
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
            executable='/bin/bash',
            text=True,
            timeout=600,
            cwd="/home",
            env=os.environ.copy()
        )
        if result.returncode == 0:
            return f"Command successed:\n{result.stdout}"
        return f"Command failed (exit code {result.returncode}):\n{result.stderr}\ncommand:\n{command}"
    except subprocess.TimeoutExpired:
        return f"Command timed out after 600 seconds"
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
            handler: Callable[[ToolCallRequest], ToolMessage | Command]
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
                tool_call_id=request.tool_call["id"]
                )


import multiprocessing as mp
from functools import wraps

def run_with_timeout(func, args, kwargs, timeout):
    """Run function in separate process with timeout"""
    def wrapper(queue, func, args, kwargs):
        try:
            result = func(*args, **kwargs)
            queue.put(('success', result))
        except Exception as e:
            queue.put(('error', e))
    
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
        if status == 'error':
            raise result
        return result
    
    raise RuntimeError("Process terminated without result")

@wrap_tool_call
def tools_time_limit(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    tool_timeout = request.runtime.context["tool_timeout"]
    start_time = time.monotonic()

    try:
        tool_name = request.tool_call["name"]
        
        if tool_name == "manage_todo":
            return handler(request)

        result = run_with_timeout(handler, (request,), {}, tool_timeout)
        return result
        
    except TimeoutError as e:
        return ToolMessage(
            content=f"Error: Tool execution timed out after {tool_timeout} seconds",
            tool_call_id=request.tool_call["id"]
        )
    except Exception as e:
        elapsed = time.monotonic() - start_time
        return ToolMessage(
            content=f"Tool '{request.tool_call["name"]}' failed after {elapsed:.2f}s: {e}",
            tool_call_id=request.tool_call["id"]
            )
        
    


def format_time(time_in_sec: int):
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {int(time_in_sec % 60)}secs"

@before_model(can_jump_to=["end"])
def agent_limitations(state: AgentState, runtime: Runtime) -> dict[str|Any] | None:
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
                "jump_to": "end"
            }
    if steps_remaining == 1:
        return {"messages": [{
            "role": "system", 
            "content": "This is your last answer, make sure submission code is available."}]}

    time_remaining = time_remaining - (time.monotonic() - step_time)
    reminder_msg = {"role": "system", "content": f"<TOTAL TIME REMAINING: {format_time(time_remaining)}>\n<TOTAL STEPS REMAINING: {steps_remaining}>"}
    runtime.context["step_time"] = time.monotonic()
    runtime.context["time_remaining"] = time_remaining
    return {"messages": [reminder_msg]}


from json import JSONDecodeError
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
            request.messages = request.messages + [{"role": "user", "content": f"Error: {e}"}]


TODO_SYSTEM_PROMPT = """"You are an autonomous ML Agent solving competitive ML tasks.

EXAMPLE WORKFLOW FOR A CLASSIFICATION TASK:

Step 1: Analysis
- Read task_description.txt
- Load and inspect train.csv, test.csv  
- Identify target column and metric

Step 2: Baseline
- Create baseline.py with LogisticRegression
- Implement cross-validation
- Run and record score: 0.75

Step 3: Improve
- Try RandomForest → score: 0.78 ✓ (keep)
- Add feature scaling → score: 0.79 ✓ (keep)
- Try XGBoost → score: 0.82 ✓ (keep)

Step 4: Submit
- Generate predictions on test.csv
- Save submission.csv
- Verify format

NOW CREATE YOUR PLAN FOLLOWING THIS STRUCTURE:

## Core Principles
- Work INDEPENDENTLY without asking user questions
- Create at least one step for each workflow item 
- Always TEST your solutions and iterate based on metrics
- Continuously IMPROVE until you achieve the best possible score

## Standard Workflow

### 1. Analysis & Planning (Initial Tasks)
- [ ] Read and analyze the task description thoroughly
- [ ] Examine the data structure and features
- [ ] Identify the target metric and evaluation approach
- [ ] Create a baseline solution plan
- [ ] List potential improvement strategies

### 2. Baseline Implementation
- [ ] Implement a simple baseline model (e.g., LogisticRegression, RandomForest, ResNet)
- [ ] Set up proper train/validation split or cross-validation
- [ ] Create evaluation script that calculates the target metric
- [ ] Run baseline and record the score

### 3. Iterative Improvement Cycle
Repeat until metric plateaus or time constraints are met:
- [ ] Analyze current model's weaknesses (feature importance, error analysis)
- [ ] Implement ONE improvement at a time:
  * Feature engineering (scaling, encoding, new features)
  * Try different algorithms (XGBoost, CatBoost)
  * Hyperparameter tuning (GridSearch, RandomSearch, Optuna)
  * Ensemble methods (stacking, blending, voting)
  * Handle class imbalance (SMOTE, class_weight)
- [ ] Test the change and compare metric with previous best
- [ ] Keep the change if it improves the metric, discard otherwise
- [ ] Document the best score achieved so far

### 4. Final Solution
- [ ] Train final model on all available training data
- [ ] Generate predictions on test set
- [ ] Save submission file in correct format
- [ ] Verify submission file format matches requirements

## Important Rules
- NEVER ask user for clarification - make reasonable assumptions
- ALWAYS run your code after writing it to verify it works
- ALWAYS evaluate metric after each significant change
- Keep track of best score and corresponding approach
- If stuck, try a different algorithm or feature engineering approach
- Read error messages carefully and fix issues immediately

## Metric Tracking
After each experiment, update your progress:
Best Score: [current_best]
Approach: [brief_description_of_best_approach]
"""


import json
from datetime import datetime


_todo_state = {
    "tasks": [],
    "completed": [],
    "created_at": None
}

@tool
def manage_todo(
    action: str,
    task: str = None,
    task_id: int = None
) -> str:
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
    
    if action == 'create':
        if not task:
            return "Error: 'create' requires 'task' parameter with your plan"
        
        # Parse tasks from string
        lines = [line.strip() for line in task.split('\n') if line.strip()]
        tasks = []
        for line in lines:
            # Remove numbering like "1.", "- ", etc
            cleaned = re.sub(r'^[\d\.\-\*\[\]\s]+', '', line)
            if cleaned:
                tasks.append(cleaned)
        
        _todo_state = {
            "tasks": tasks,
            "completed": [False] * len(tasks),
            "created_at": datetime.now().isoformat()
        }
        
        return (
            f"✓ TODO list created with {len(tasks)} tasks:\n\n"
            + "\n".join(f"{i+1}. [ ] {t}" for i, t in enumerate(tasks))
            + "\n\nNow start working on task 1!"
        )
    
    if action == 'add':
        if not task:
            return "Error: 'add' requires 'task' parameter"
        
        if not _todo_state["created_at"]:
            return "Error: TODO list not initialized. Use action='create' first!"
        
        _todo_state["tasks"].append(task)
        _todo_state["completed"].append(False)
        task_num = len(_todo_state["tasks"])
        
        return f"✓ Added task #{task_num}: {task}"
    
    if action == 'complete':
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
            f"({100*completed_count//total_count}%)"
        )
    
    if action == 'view':
        if not _todo_state["created_at"]:
            return "No TODO list exists. Create one with action='create'!"
        
        output = ["Current TODO List:", ""]
        for i, (task, done) in enumerate(zip(_todo_state["tasks"], _todo_state["completed"])):
            status = "✓" if done else " "
            output.append(f"{i+1}. [{status}] {task}")
        
        completed = sum(_todo_state["completed"])
        total = len(_todo_state["tasks"])
        output.append("")
        output.append(f"Progress: {completed}/{total} completed ({100*completed//total}%)")
        
        return "\n".join(output)
    
    if action == 'progress':
        if not _todo_state["created_at"]:
            return "No TODO list exists yet"
        
        completed = sum(_todo_state["completed"])
        total = len(_todo_state["tasks"])
        pct = 100 * completed // total if total > 0 else 0
        
        next_task = None
        for i, (task, done) in enumerate(zip(_todo_state["tasks"], _todo_state["completed"])):
            if not done:
                next_task = (i+1, task)
                break
        
        result = [
            f"Progress: {completed}/{total} tasks ({pct}%)",
            f"Completed: {completed}",
            f"Remaining: {total - completed}"
        ]
        
        if next_task:
            result.append(f"\nNext task: #{next_task[0]} - {next_task[1]}")
        else:
            result.append("\n🎉 All tasks completed!")
        
        return "\n".join(result)
    
    return f"Error: Unknown action '{action}'. Valid: create, add, complete, view, progress"

@after_agent(can_jump_to="model")
def check_todo(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    global _todo_state
    steps_remaining = runtime.context.get("steps_remaining")
    if len(_todo_state["completed"]) == 0 and steps_remaining > 0:
        return {
            "messages": [{"role": "system", "content": "Todo not initialized, initialize and complete it before finish"}],
            "jump_to": "model"
        }
    if not all(_todo_state["completed"]) and steps_remaining > 0:
        return {
            "messages": [{"role": "system", "content": f"Not all todo tasks are completed, complete tasks before finish:\n{manage_todo.run({"action": "view"})}"}],
            "jump_to": "model"
        }


# class TodoEnforcementMiddleware(AgentMiddleware):
#     """Prevent completing multiple TODO tasks in one step"""
    
#     def __init__(self):
#         super().__init__()
#         self.todos_completed_this_step = 0
#         self.complited_id = None
    
#     def before_model(self, state: AgentState, runtime: Runtime):
#         """Reset counter before model call"""
#         self.todos_completed_this_step = 0
#         self.complited_id = None
#         return None
    
#     def wrap_tool_call(
#         self,
#         request: ToolCallRequest,
#         handler: Callable[[ToolCallRequest], ToolMessage | Command]
#     ) -> ToolMessage | Command:
#         tool_name = request.tool_call["name"]
        
#         if tool_name == "manage_todo":
#             args = request.tool_call["args"]
#             if args.get("action") == "complete":
#                 if self.todos_completed_this_step > 0:
#                     return ToolMessage(
#                         content=(
#                             "❌ Error: You can only complete ONE TODO task per step!\n"
#                             f"You already completed a task with id: {self.complited_id} in this step.\n"
#                             "Wait for the next turn to complete the next task.\n"
#                             "This ensures you actually do the work between completions."
#                         ),
#                         tool_call_id=request.tool_call["id"]
#                     )
                
#                 # Вызываем handler
#                 result = handler(request)
                
#                 # Если успешно, увеличиваем счетчик
#                 if isinstance(result, ToolMessage) and "✓ Completed" in result.content:
#                     self.complited_id = args.get("task_id")
#                     self.todos_completed_this_step += 1
                
#                 return result
        
#         # Для других инструментов - просто вызываем handler
#         return handler(request)




SYSTEM_PROMPT = """You are an ML Agent solving competitive tasks.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MANDATORY FIRST ACTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before doing ANYTHING else, you MUST call manage_todo(action="create", task="...")
to create your work plan.

**You may customize the plan based on the specific task, BUT it must include
ALL these stages (minimum 7 steps, you can add more details):**

1. Read and analyze task description/requirements
2. Explore data structure and characteristics  
3. Create baseline model
4. Evaluate baseline score
5. Improve model iteratively (may split into multiple steps)
6. Generate final submission file
7. Validate submission format and readiness

**Examples of good customization:**

For computer vision task:
1. Read task description and evaluation metric
2. Explore image data (shape, distribution, class balance)
3. Check for data augmentation opportunities
4. Create CNN baseline (ResNet18)
5. Evaluate baseline on validation set
6. Try advanced architectures (EfficientNet, ViT)
7. Optimize hyperparameters (lr, batch_size, augmentation)
8. Ensemble top models
9. Generate submission predictions
10. Validate output format matches requirements

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Workflow Rules:

1. **Start**: Call manage_todo(action="create") with your full plan
2. **During work**: 
   - Call manage_todo(action="complete", task_id=X) after finishing each task
   - Call manage_todo(action="progress") to check what's next
3. **Before finishing**: Ensure all tasks are completed

## Best Practices:

- Test every script you create
- Track your best score after each model
- If stuck, try different approach
- NEVER finish without submission file

## Example Flow:

User: [gives task]
You: manage_todo(action="create", task="1. Read data\\n2. Baseline\\n3. Improve\\n4. Submit")
→ create_file("explore.py", ...)
→ run_python("explore.py")
→ manage_todo(action="complete", task_id=1)
→ create_file("baseline.py", ...)
→ [continue...]
"""

import time
from langchain_core.callbacks import UsageMetadataCallbackHandler
def run_react(cfg: Config):
    code_dir, submission_dir, logs_dir = prepare_workflow(cfg)
    with open(cfg.description, "r", encoding="utf-8") as td:
        task_description = td.read()
    llm = get_llm_model(cfg.provider, cfg.model, cfg.temperature)
    agent = create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT,
        tools=[create_file, run_command, read_file, run_python, list_directory, manage_todo],
        middleware=[
            # TodoListMiddleware(
            #     system_prompt=TODO_SYSTEM_PROMPT
            # ),
            # LLMToolSelectorMiddleware(
            #     model=llm,
            #     max_tools=1
            # ),
            agent_limitations,
            # TodoEnforcementMiddleware(),
            ToolCycleMiddleware(),
            tools_time_limit,
            handle_parse_errors,
            SummarizationMiddleware(
                model=llm,
                trigger=("tokens", 20000),
                keep=("messages", 15)
            ),
            check_todo
        ]
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
            "max_steps": cfg.max_steps
            },
            config={"callbacks": [callback]}
    ):
        with capture_stdout() as output:
            step["messages"][-1].pretty_print()
        logger.info(output.getvalue())
    with open(logs_dir / "usage_metadata.json", "w", encoding="utf-8") as f:
        json.dump(callback.usage_metadata, f, indent=2)
