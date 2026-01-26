from pathlib import Path
from typing import Literal
import subprocess
import os
from langchain_core.tools import tool
from loguru import logger
import shutil


def create_file_tools(data_dir: Path, code_dir: Path, submission_dir: Path):
    
    @tool
    def list_files(directory: Literal["data", "code", "submission"], pattern: str | None = None) -> str:
        """List files in the specified directory.
        
        Args:
            directory: Which directory to list:
                - 'data': Dataset files (read-only)
                - 'code': Your working directory for experiments
                - 'submission': Final submission directory
            pattern: Optional glob pattern (e.g., '*.csv', '*.py')
        
        Returns:
            List of files with their sizes.
        """
        try:
            dir_map = {"data": data_dir, "code": code_dir, "submission": submission_dir}
            target_dir = dir_map[directory]
            
            if pattern:
                files = list(target_dir.glob(pattern))
            else:
                files = list(target_dir.iterdir())
            
            if not files:
                return f"No files found in {directory}/"
            
            result = [f"Files in {directory}/:"]
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
            filepath: Path to file. Prefix determines location:
                - 'data/file.csv' → reads from data directory
                - 'submission/submission.py' → reads from submission directory
                - 'file.py' → reads from code directory (default)
            start_line: Optional starting line number (1-indexed)
            end_line: Optional ending line number (1-indexed)
        
        Returns:
            File contents or specified line range.
            
        Note:
            Large files (>50000 chars) are automatically truncated.
            Use start_line/end_line to read specific sections.
        """
        try:
            if filepath.startswith("data/"):
                full_path = data_dir / filepath.replace("data/", "")
            elif filepath.startswith("submission/"):
                full_path = submission_dir / filepath.replace("submission/", "")
            else:
                full_path = code_dir / filepath
            
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
            filepath: Path relative to workspace root. Examples:
                    - 'code/train.py' - script in code directory
                    - 'submission/submission.csv' - submission file
            content: File content to write
            
        Returns:
            Success message with file size and absolute path
            
        Important:
            - For code: filepath should be 'code/filename.py'
            - For submissions: filepath should be 'submission/filename'
            - DO NOT include working directory prefix (it's added automatically)
            
        Example:
            write_file('code/train.py', 'import pandas as pd\\n...')
            write_file('submission/solution.csv', 'id,prediction\\n1,0.5\\n...')
        """
        try:
            workspace_root = code_dir.parent
            
            filepath = filepath.lstrip('/')
            
            if filepath.startswith("submission/"):
                target_path = workspace_root / filepath
                rel_path = filepath[len("submission/"):]
                logger.info(f"Writing to submission: {rel_path}")
            elif filepath.startswith("code/"):
                target_path = workspace_root / filepath
                rel_path = filepath[len("code/"):]
                logger.info(f"Writing to code: {rel_path}")
            else:
                target_path = code_dir / filepath
                logger.info(f"Writing to code (no prefix): {filepath}")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            target_path.write_text(content, encoding='utf-8')
            
            size = len(content.encode('utf-8'))
            return (f"Successfully wrote {size:,} bytes to {filepath}\n"
                    f"Absolute path: {target_path.absolute()}")
            
        except Exception as e:
            logger.error(f"Error writing file {filepath}: {e}")
            return f"Error writing file: {str(e)}"
    
    @tool
    def edit_file(filepath: str, old_content: str, new_content: str) -> str:
        """Edit a specific part of an existing file by replacing exact text.
        
        Args:
            filepath: Path to file (supports 'submission/' prefix)
            old_content: Exact text to find and replace
            new_content: New text to insert
        
        Returns:
            Success message or error if old_content not found.
            
        Note:
            old_content must appear exactly once. If it appears multiple times,
            make it more specific by including more surrounding context.
        """
        try:
            if filepath.startswith("submission/"):
                full_path = submission_dir / filepath.replace("submission/", "")
            elif filepath.startswith("data/"):
                return "Error: Cannot edit data/ directory files (read-only)"
            else:
                full_path = code_dir / filepath
            
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
            
            return f"Successfully edited {filepath}: replaced {len(old_content)} chars with {len(new_content)} chars"
        except Exception as e:
            logger.error(f"Error editing file {filepath}: {e}")
            return f"Error editing file: {str(e)}"
    
    @tool
    def execute_python(script_path: str, script_args: list[str] | None = None) -> str:
        """Execute a Python script from the code/ directory.
        
        Args:
            script_path: Path to the script relative to code/ (e.g., 'train.py'). 
                        **Do not include 'code/' prefix.**
            script_args: Optional list of command-line arguments to pass to the script.
        
        Returns:
            Output of the script (stdout and stderr) and exit code.
            
        Examples:
            execute_python("train.py")  # Run training script
            execute_python("evaluate.py")  # Run evaluation
            execute_python("preprocess.py", ["--verbose"])  # Run with arguments
        
        Notes:
            - The script runs in code/ directory, so use '../data/' to access dataset files.
            - The code/ prefix is not needed; provide paths relative to code/.
            - Use '../submission/' to access files in the submission directory if needed.
        """
        try:
            full_path = code_dir / script_path
            
            if not full_path.exists():
                return f"Error: Script '{script_path}' does not exist in code/ directory"
            
            cmd = ["uv", "run", str(full_path)]
            if script_args:
                cmd.extend(script_args)
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(code_dir),
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
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
            logger.error(f"Error executing script {script_path}: {e}")
            return f"Error executing script: {str(e)}"
    
    @tool
    def execute_shell(command: str) -> str:
        """Execute a shell command in the code directory.
        
        SAFETY: Dangerous commands (rm, sudo, wget, etc.) are blocked.
        
        Args:
            command: Shell command to execute
        
        Returns:
            Command output (stdout and stderr)
            
        Examples:
            execute_shell("pip install lightgbm") - install package
            execute_shell("pip list | grep pandas") - check packages
            execute_shell("head -n 5 ../data/train.csv") - preview data
            execute_shell("wc -l ../data/train.csv") - count lines
            execute_shell("ls -lh ../submission/") - check submission files
        
        Blocked Commands (for safety):
            rm, rmdir, sudo, su, chmod, chown, wget, curl, dd, mkfs, kill, mv, cp
        
        Note:
            - Use '../data/' to access data files
            - Use '../submission/' to access submission files
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
                cwd=str(code_dir),
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            
            output = []
            output.append(f"Command: {command}")
            output.append(f"Working directory: code/")
            
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
    def install_packages(
        packages: list[str], 
        method: Literal["quick", "requirements", "project"] = "quick"
    ) -> str:
        """Install Python packages using UV (flexible method selection).
        
        Args:
            packages: List of package specifications (e.g., ["pandas==2.0.0", "xgboost"])
            method: Installation method
                - "quick": Fast install with 'uv pip install' (default)
                - "requirements": Create requirements.txt then install
                - "project": Create pyproject.toml project (most robust)
        
        Returns:
            Installation status
            
        Examples:
            # Quick install (most common)
            install_packages(["pandas", "scikit-learn", "xgboost"])
            
            # Create requirements.txt for reproducibility
            install_packages(
                ["pandas==2.0.0", "scikit-learn>=1.3.0"],
                method="requirements"
            )
            
            # Full project setup (professional)
            install_packages(
                ["pandas", "xgboost", "lightgbm"],
                method="project"
            )
        
        Common packages:
            Data: pandas, numpy, polars
            ML: scikit-learn, xgboost, lightgbm, catboost
            DL: torch, tensorflow
            Utils: tqdm, joblib, matplotlib, seaborn
        """
        import subprocess
        
        try:
            if not shutil.which("uv"):
                return (
                    "UV not installed\n"
                    "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
                )
            
            if not packages:
                return "No packages specified"
            
            logger.info(f"Installing {len(packages)} packages via method '{method}'")
            
            if method == "quick":
                cmd = ["uv", "pip", "install"] + packages
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(code_dir),
                )
                
                output = []
                if result.returncode == 0:
                    output.append("PACKAGES INSTALLED\n")
                    output.append(f"Method: uv pip install")
                    output.append(f"Packages ({len(packages)}):")
                    for pkg in packages:
                        output.append(f"   {pkg}")
                    
                    if "Installed" in result.stdout:
                        installed = [
                            line for line in result.stdout.split('\n')
                            if 'installed' in line.lower()
                        ][:10]
                        if installed:
                            output.append("\nInstalled versions:")
                            for line in installed:
                                output.append(f"  {line.strip()}")
                else:
                    output.append("❌ INSTALLATION FAILED\n")
                    output.append(result.stderr[:1000])
                
                return "\n".join(output)
            
            elif method == "requirements":
                req_path = code_dir / "requirements.txt"
                req_path.write_text("\n".join(packages))
                
                output = [f"Created requirements.txt ({len(packages)} packages)\n"]
                
                cmd = ["uv", "pip", "install", "-r", "requirements.txt"]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(code_dir)
                )
                
                if result.returncode == 0:
                    output.append("Installed from requirements.txt")
                else:
                    output.append(f"Installation failed:\n{result.stderr[:500]}")
                
                return "\n".join(output)
            
            elif method == "project":
                pyproject_path = code_dir / "pyproject.toml"
                
                content = f'''[project]
    name = "ml-solution"
    version = "0.1.0"
    requires-python = ">=3.9"
    dependencies = [
    '''
                for pkg in packages:
                    content += f'    "{pkg}",\n'
                content += ''']

    [build-system]
    requires = ["hatchling"]
    build-backend = "hatchling.build"
    '''
                
                pyproject_path.write_text(content)
                
                output = ["Created pyproject.toml\n"]
                
                cmd = ["uv", "sync"]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(code_dir)
                )
                
                if result.returncode == 0:
                    output.append("Project synced successfully")
                    output.append("Use 'uv run <script>' to execute")
                else:
                    output.append(f"Sync failed:\n{result.stderr[:500]}")
                
                return "\n".join(output)
            
            else:
                return f"Unknown method: {method}"
            
        except subprocess.TimeoutExpired:
            return "Installation timed out (600s)"
        except Exception as e:
            logger.error(f"Install error: {e}")
            return f"Error: {str(e)}"

    @tool
    def finish() -> None:
        """Signal task completion and provide final summary.
        
        Call this tool when you are satisfied with your solution.
        """
        return None

    return [list_files, read_file, write_file, edit_file, 
            execute_python, execute_shell, install_packages, finish]


def get_tools(data_dir: Path, code_dir: Path, submission_dir: Path) -> dict:
    """Create all tools for the agent.
    
    Args:
        data_dir: Path to dataset directory (read-only)
        code_dir: Path to working directory for experiments
        submission_dir: Path to submission directory for final files
    
    Returns:
        Dictionary mapping tool names to tool functions
    """
    tools = create_file_tools(data_dir, code_dir, submission_dir)
    return {tool.name: tool for tool in tools}