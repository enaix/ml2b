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

from .prompts import SYSTEM_PROMPT
from .router import get_llm_model
from ..config import Config
from .tools import get_tools


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
    KEEP_RECENT = 10
    
    total_messages = len(messages)
    system_count = sum(1 for msg in messages if msg.type == "system")
    
    if total_messages < SUMMARIZE_THRESHOLD:
        return {}
    
    if total_messages - system_count < SUMMARIZE_THRESHOLD - system_count:
        return {}
    
    system_indices = []
    for i, msg in enumerate(messages):
        if msg.type == "system":
            system_indices.append(i)
    
    first_non_system_idx = max(system_indices) + 1 if system_indices else 0
    
    system_messages = messages[:first_non_system_idx]
    non_system_messages = messages[first_non_system_idx:]
    
    if len(non_system_messages) < KEEP_RECENT:
        return {}
    
    recent_messages = non_system_messages[-KEEP_RECENT:]
    messages_to_summarize = non_system_messages[:-KEEP_RECENT]
    
    summary_content = []
    for msg in messages_to_summarize:
        if hasattr(msg, 'content'):
            content_str = str(msg.content)[:500]
            summary_content.append(f"{msg.type}: {content_str}")
    
    summary_prompt = f"""Summarize the following agent conversation history concisely, preserving:
1. Key decisions and actions taken
2. Important findings and results
3. Current progress on the task
4. Any critical errors or issues encountered

History to summarize:
{chr(10).join(summary_content)}

Provide a concise summary (max 400 tokens)."""
    
    chat_model = runtime.context.chat_model
    summary_response = await chat_model.ainvoke([
        {"role": "user", "content": summary_prompt}
    ])
    
    summary_message = AIMessage(
        content=f"[HISTORY SUMMARY]\n{summary_response.content}"
    )
    
    new_messages = [*system_messages, summary_message, *recent_messages]
    
    logger.info(f"Summarized {len(messages_to_summarize)} messages. New history: {len(new_messages)} messages (including {len(system_messages)} system messages)")
    return {"messages": new_messages}


async def progress_reminder(state: AgentState) -> dict:
    iterations = state.iterations
    remaining = state.remaining_steps
    
    if iterations % 5 == 0 or remaining <= 3:
        if remaining <= 2:
            msg = (
                f"CRITICAL: Only {remaining} steps remaining!\n"
                f"You MUST finalize your submission NOW.\n"
                f"Use create_submission() tool to save your best solution to submission/ directory."
            )
        elif remaining <= 5:
            msg = (
                f"WARNING: {remaining} steps remaining.\n"
                f"Focus on creating and testing your final submission."
            )
        else:
            msg = f"Progress: {iterations} iterations completed, {remaining} steps remaining."
        
        return {"messages": [{"role": "user", "content": msg}]}
    
    return {}


async def submission_validator(state: AgentState, runtime: Runtime[ContextSchema]) -> dict:
    submission_dir = runtime.context.submission_dir
    submission_file = submission_dir / "submission.py"
    
    if not submission_file.exists():
        return {
            "messages": [{
                "role": "user", 
                "content": (
                    "ERROR: No submission.py file found in submission/ directory.\n"
                    "You MUST create a submission file using create_submission() tool.\n"
                    "The file must contain functions: train(), prepare_val(), predict(), run()"
                )
            }],
            "submission_ready": False
        }
    
    
    try:
        with open(submission_file, 'r') as f:
            content = f.read()
        
        required_functions = ['train', 'prepare_val', 'predict', 'run']
        missing_functions = [func for func in required_functions if f"def {func}(" not in content]
        
        if missing_functions:
            return {
                "messages": [{
                    "role": "user",
                    "content": (
                        f"ERROR: Missing required functions in {submission_file.name}: {missing_functions}\n"
                        "All four functions must be present: train(), prepare_val(), predict(), run()\n"
                        "Fix your submission file."
                    )
                }],
                "submission_ready": False
            }
        
        lines = content.split('\n')
        global_assignments = []
        in_function = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def ') or stripped.startswith('class '):
                in_function = True
            elif not in_function and '=' in stripped and not stripped.startswith('import') and not stripped.startswith('#'):
                if stripped and not stripped.startswith(('from ', 'import ')):
                    global_assignments.append(stripped[:50])
        
        if global_assignments:
            logger.warning(f"Potential global variables detected: {global_assignments}")
            return {
                "messages": [{
                    "role": "user",
                    "content": (
                        f"WARNING: Possible global variables detected in {submission_file.name}:\n"
                        f"{global_assignments[:3]}\n"
                        "CRITICAL CONSTRAINT: NO GLOBAL VARIABLES allowed!\n"
                        "All code must be inside functions or classes."
                    )
                }],
                "submission_ready": False
            }
        
        logger.info(f"Submission validated: {submission_file.name}")
        return {
            "messages": [{
                "role": "user",
                "content": (
                    f"Submission validated successfully!\n"
                    f"File: {submission_file.name} ({submission_file.stat().st_size} bytes)\n"
                    f"All required functions present: {required_functions}\n"
                    f"No global variables detected.\n\n"
                    f"You can now finish, or continue improving if steps remain."
                )
            }],
            "submission_ready": True
        }
        
    except Exception as e:
        logger.error(f"Error validating submission: {e}")
        return {
            "messages": [{
                "role": "user",
                "content": f"ERROR: Failed to validate submission: {str(e)}"
            }],
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
    
    if state.remaining_steps <= 0:
        logger.info("No steps remaining, ending")
        return "__end__"
    
    last_message = state.messages[-1]

    if last_message.tool_calls:
        for tc in last_message.tool_calls:
            if tc['name'] == 'finish':
                logger.info("Finish tool called, ending workflow")
                return "__end__"

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    if state.remaining_steps <= 3 or (state.iterations > 0 and state.iterations % 10 == 0):
        logger.info("Triggering submission validation")
        return "validate"
    
    return "progress"


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
    
    builder.add_edge("execute_tools", "cleanup_messages")
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
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system", 
            "content": (
                f"TASK INSTRUCTIONS:\n{cfg.description}\n\n"
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
    
    return code_dir, submission_dir


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