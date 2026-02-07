import argparse
import os
from pathlib import Path
from  dotenv import load_dotenv
from .config import Config
from src.agent.agent import run_agent, run_react


def file_path(p: str):
    path = Path(p).resolve()
    if path.is_file():
        return path
    else:
        raise FileNotFoundError


def get_args():
    parser = argparse.ArgumentParser()
    cwd = Path(os.getcwd()).resolve()
    parser.add_argument("--model", help="Agent model name")
    parser.add_argument("-p", "--provider", help="Agent model name", default="openai", choices=["openai", "vertex"])
    parser.add_argument("--data-dir", help="Path to data directory")
    parser.add_argument("--description", help="Path to description file")
    parser.add_argument("--env-file", type=file_path, default=cwd / ".env", help="Path to .env file with credentials")
    parser.add_argument("--working-dir", default=cwd / "working", help="Path to agent working directory")
    parser.add_argument("--exp-name", default="agent", help="Experiment name")
    parser.add_argument("-t", "--temperature", default=.0, type=float, help="LLM temperature")
    parser.add_argument("--max-steps", default=50, type=int, help="Maximum agent workflow steps")
    parser.add_argument("--tool-timeout", default=1800, type=int, help="Tools call execution timeout")
    parser.add_argument("--time-limit", default=10800, type=int, help="Agent working time limit")
    args = parser.parse_args()
    args = Config.model_validate(vars(args))
    return args


def main():
    args = get_args()
    load_dotenv(dotenv_path=args.env_file)
    # run_agent(args)
    run_react(args)


if __name__ == "__main__":
    main()