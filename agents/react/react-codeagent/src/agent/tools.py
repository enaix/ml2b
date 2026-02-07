from pathlib import Path
from typing import Literal
import subprocess
import os
from langchain_core.tools import tool
from loguru import logger
import shutil


def create_file_tools():
    """Создаём инструменты БЕЗ захардкоженных путей.
    Пути будут передаваться через аргументы или environment."""
    
    @tool
    def list_files(directory: str, pattern: str | None = None) -> str:
        """List files in a directory.
        
        Args:
            directory: Absolute path to directory (e.g., '/home/agent/working/bench/data')
            pattern: Optional glob pattern (e.g., '*.csv', '*.py')
        
        Returns:
            List of files with their sizes.
            
        Examples:
            list_files('/home/agent/working/bench/data')
            list_files('/home/agent/working/bench/code', '*.py')
        """
        try:
            target_dir = Path(directory)
            
            if not target_dir.exists():
                return f"Error: Directory '{directory}' does not exist"
            
            if pattern:
                files = list(target_dir.glob(pattern))
            else:
                files = list(target_dir.iterdir())
            
            if not files:
                return f"No files found in {directory}"
            
            result = [f"Files in {directory}:"]
            for f in sorted(files):
                if f.is_file():
                    size = f.stat().st_size
                    size_str = f"{size:,} bytes" if size < 1024*1024 else f"{size/(1024*1024):.2f} MB"
                    result.append(f"  - {f.name} ({size_str})")
                elif f.is_dir():
                    result.append(f"  - {f.name}/ (directory)")
            
            return "\n".join(result)
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return f"Error listing files: {str(e)}"
    
    @tool
    def read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str:
        """Read contents of a file.
        
        Args:
            filepath: Absolute path to file (e.g., '/home/agent/working/bench/data/train.csv')
            start_line: Optional starting line number (1-indexed)
            end_line: Optional ending line number (1-indexed)
        
        Returns:
            File contents or specified line range.
            
        Note:
            Large files (>50000 chars) are automatically truncated.
            Use start_line/end_line to read specific sections.
        """
        try:
            full_path = Path(filepath)
            
            if not full_path.exists():
                return f"Error: File '{filepath}' does not exist"
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                if start_line is not None or end_line is not None:
                    lines = f.readlines()
                    start = (start_line - 1) if start_line else 0
                    end = end_line if end_line else len(lines)
                    content = "".join(lines[start:end])
                    return f"Lines {start+1}-{end} of {filepath}:\n{content}"
                else:
                    content = f.read()
                    if len(content) > 50000:
                        return f"File is large ({len(content)} chars). First 50000 characters:\n{content[:50000]}\n\n[... truncated, use start_line/end_line to read specific parts]"
                    return content
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            return f"Error reading file: {str(e)}"
    
    @tool
    def write_file(filepath: str, content: str) -> str:
        """Create or overwrite a file with specified content.
        
        Args:
            filepath: Absolute path to file (e.g., '/home/agent/working/bench/code/train.py')
            content: File content to write
            
        Returns:
            Success message with file size
            
        Examples:
            write_file('/home/agent/working/bench/code/train.py', 'import pandas as pd\\n...')
            write_file('/home/agent/working/bench/submission/submission.py', '...')
        """
        try:
            target_path = Path(filepath)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            target_path.write_text(content, encoding='utf-8')
            
            size = len(content.encode('utf-8'))
            return (f"✓ Wrote {size:,} bytes to {target_path.name}\n"
                    f"Path: {target_path.absolute()}")
            
        except Exception as e:
            logger.error(f"Error writing file {filepath}: {e}")
            return f"Error writing file: {str(e)}"
    
    @tool
    def edit_file(filepath: str, old_content: str, new_content: str) -> str:
        """Edit a file by replacing exact text.
        
        Args:
            filepath: Absolute path to file
            old_content: Exact text to find and replace
            new_content: New text to insert
        
        Returns:
            Success message or error if old_content not found.
            
        Note:
            old_content must appear exactly once. If it appears multiple times,
            make it more specific by including more surrounding context.
        """
        try:
            full_path = Path(filepath)
            
            if not full_path.exists():
                return f"Error: File '{filepath}' does not exist. Use write_file to create it first."
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if old_content not in content:
                return f"Error: old_content not found in {filepath}"
            
            count = content.count(old_content)
            if count > 1:
                return f"Error: old_content appears {count} times. Make it more specific."
            
            new_file_content = content.replace(old_content, new_content, 1)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_file_content)
            
            return f"✓ Edited {full_path.name}: replaced {len(old_content)} chars with {len(new_content)} chars"
        except Exception as e:
            logger.error(f"Error editing file {filepath}: {e}")
            return f"Error editing file: {str(e)}"
    
    @tool
    def execute_python(script_path: str, working_dir: str, script_args: list[str] | None = None) -> str:
        """Execute a Python script.
        
        Args:
            script_path: Absolute path to script (e.g., '/home/agent/working/bench/code/train.py')
            working_dir: Directory to run from (e.g., '/home/agent/working/bench/code')
            script_args: Optional command-line arguments
        
        Returns:
            Script output (stdout and stderr) and exit code.
            
        Examples:
            execute_python('/home/agent/.../code/train.py', '/home/agent/.../code')
            execute_python('/home/agent/.../code/eval.py', '/home/agent/.../code', ['--verbose'])
        """
        try:
            script_path = Path(script_path)
            working_dir = Path(working_dir)
            
            if not script_path.exists():
                return f"Error: Script '{script_path}' does not exist"
            
            # Находим uv или используем python
            uv_path = shutil.which("uv")
            if uv_path:
                cmd = [uv_path, "run", "python", str(script_path)]
            else:
                cmd = ["python", str(script_path)]
            
            if script_args:
                cmd.extend(script_args)
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(working_dir),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                timeout=600
            )
            
            output = []
            if result.returncode == 0:
                output.append(f"✓ Script executed successfully (exit code 0)")
            else:
                output.append(f"✗ Script failed with exit code {result.returncode}")
            
            if result.stdout:
                output.append("\n--- STDOUT ---")
                stdout = result.stdout
                if len(stdout) > 5000:
                    output.append(stdout[:5000])
                    output.append(f"\n[... truncated {len(stdout) - 5000} characters ...]")
                else:
                    output.append(stdout)
            
            if result.stderr:
                output.append("\n--- STDERR ---")
                stderr = result.stderr
                if len(stderr) > 5000:
                    output.append(stderr[:5000])
                    output.append(f"\n[... truncated {len(stderr) - 5000} characters ...]")
                else:
                    output.append(stderr)
            
            return "\n".join(output)
            
        except subprocess.TimeoutExpired:
            return f"Error: Script execution timed out after 600 seconds"
        except Exception as e:
            logger.error(f"Error executing script: {e}")
            return f"Error executing script: {str(e)}"
    
    @tool
    def execute_shell(command: str, working_dir: str) -> str:
        """Execute a shell command.
        
        Args:
            command: Shell command to execute
            working_dir: Directory to run from (e.g., '/home/agent/working/bench/code')
        
        Returns:
            Command output (stdout and stderr)
            
        Examples:
            execute_shell("pip install pandas", "/home/agent/.../code")
            execute_shell("ls -lh", "/home/agent/.../data")
        
        Blocked Commands (for safety):
            rm, rmdir, sudo, su, chmod, chown, wget, curl, dd, mkfs, kill, mv, cp
        """
        
        BLOCKED_COMMANDS = [
            'rm', 'rmdir', 'sudo', 'su', 'chmod', 'chown', 
            'wget', 'curl', 'dd', 'mkfs', 'kill', 'killall',
            'mv', 'cp',
        ]
        
        command_lower = command.lower()
        for blocked in BLOCKED_COMMANDS:
            if blocked in command_lower.split():
                return f"Error: Command '{blocked}' is blocked for safety"
        
        if any(pattern in command for pattern in ['$(', '`', ';', '&&', '||']):
            return "Error: Command injection patterns detected. Use simple single commands only."
        
        try:
            logger.info(f"Executing shell: {command}")
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(working_dir),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                timeout=60
            )
            
            output = []
            output.append(f"Command: {command}")
            output.append(f"Working directory: {working_dir}")
            
            if result.returncode == 0:
                output.append("✓ Exit code 0")
            else:
                output.append(f"⚠ Exit code {result.returncode}")
            
            if result.stdout:
                output.append("\n--- OUTPUT ---")
                stdout = result.stdout
                if len(stdout) > 10000:
                    output.append(stdout[:10000])
                    output.append(f"\n[... truncated {len(stdout) - 10000} characters ...]")
                else:
                    output.append(stdout)
            
            if result.stderr:
                output.append("\n--- ERRORS/WARNINGS ---")
                stderr = result.stderr
                if len(stderr) > 5000:
                    output.append(stderr[:5000])
                    output.append(f"\n[... truncated {len(stderr) - 5000} characters ...]")
                else:
                    output.append(stderr)
            
            return "\n".join(output)
            
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after 60 seconds"
        except Exception as e:
            logger.error(f"Error executing shell command: {e}")
            return f"Error executing command: {str(e)}"
    
    @tool 
    def install_packages(packages: list[str], working_dir: str) -> str:
        """Install Python packages using UV or pip.
        
        Args:
            packages: List of packages (e.g., ["pandas", "scikit-learn==1.3.0"])
            working_dir: Directory to run from (e.g., '/home/agent/working/bench/code')
        
        Returns:
            Installation status
            
        Examples:
            install_packages(["pandas", "scikit-learn", "xgboost"], "/home/agent/.../code")
        
        Common packages:
            Data: pandas, numpy, polars
            ML: scikit-learn, xgboost, lightgbm, catboost
            DL: torch, tensorflow
            Utils: tqdm, joblib, matplotlib, seaborn
        """
        try:
            if not packages:
                return "No packages specified"
            
            logger.info(f"Installing {len(packages)} packages")
            
            uv_path = shutil.which("uv")
            if uv_path:
                cmd = [uv_path, "pip", "install"] + packages
            else:
                cmd = ["pip", "install"] + packages
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(working_dir),
                timeout=600
            )
            
            output = []
            if result.returncode == 0:
                output.append("✓ PACKAGES INSTALLED\n")
                output.append(f"Packages ({len(packages)}):")
                for pkg in packages:
                    output.append(f"   {pkg}")
            else:
                output.append("❌ INSTALLATION FAILED\n")
                output.append(result.stderr[:1000])
            
            return "\n".join(output)
            
        except subprocess.TimeoutExpired:
            return "Installation timed out (600s)"
        except Exception as e:
            logger.error(f"Install error: {e}")
            return f"Error: {str(e)}"

    @tool
    def finish() -> str:
        """Signal task completion.
        
        Call this when you're satisfied with your solution.
        Make sure you've validated your submission first!
        """
        return "Task marked as complete."

    return [list_files, read_file, write_file, edit_file, 
            execute_python, execute_shell, install_packages, finish]


def get_tools(data_dir: Path, code_dir: Path, submission_dir: Path) -> dict:
    """Create tools (keeping signature for compatibility)."""
    tools = create_file_tools()
    return {tool.name: tool for tool in tools}