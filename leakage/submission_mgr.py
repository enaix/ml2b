#!/usr/bin/env python3
"""
Script to recursively find submission.py files, add path comments, and copy to output directory.
"""

import os
import ast
import argparse
from pathlib import Path


class EntrypointAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze if a file has an entrypoint and find the run function."""
    
    def __init__(self):
        self.has_main_guard = False
        self.has_top_level_executable = False
        self.run_function = None
        self.train_and_predict = None
    
    def visit_If(self, node):
        """Check for if __name__ == "__main__" pattern."""
        if (isinstance(node.test, ast.Compare) and
            isinstance(node.test.left, ast.Name) and
            node.test.left.id == "__name__" and
            len(node.test.ops) == 1 and
            isinstance(node.test.ops[0], ast.Eq) and
            len(node.test.comparators) == 1 and
            isinstance(node.test.comparators[0], ast.Constant) and
            node.test.comparators[0].value == "__main__"):
            self.has_main_guard = True
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Find the run function definition."""
        if node.name == "run":
            self.run_function = node
        elif node.name == "train_and_predict":
            self.train_and_predict = node
        # Don't traverse into function bodies for top-level executable detection
    
    def visit_Call(self, node):
        """Check for top-level function calls (executable statements)."""
        # Only count if this is at module level (col_offset == 0 for top-level)
        if hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
            # This is a heuristic - we'll check the parent context
            self.has_top_level_executable = True
        self.generic_visit(node)
    
    def visit_Expr(self, node):
        """Check for top-level expression statements."""
        if isinstance(node.value, ast.Call):
            self.has_top_level_executable = True
        self.generic_visit(node)
    
    def visit_Module(self, node):
        """Process module level statements."""
        for child in node.body:
            if isinstance(child, ast.Expr) and isinstance(child.value, ast.Call):
                self.has_top_level_executable = True
            elif isinstance(child, ast.Call):
                self.has_top_level_executable = True
        self.generic_visit(node)


def analyze_entrypoint(code):
    """
    Analyze if the code has an entrypoint and extract run function info.
    
    Returns:
        tuple: (has_entrypoint, run_function_info)
        - has_entrypoint: bool indicating if entrypoint exists
        - run_function_info: dict with function signature info or None
    """
    try:
        tree = ast.parse(code)
        analyzer = EntrypointAnalyzer()
        analyzer.visit(tree)
        
        has_entrypoint = analyzer.has_main_guard or analyzer.has_top_level_executable
        
        run_function_info = None

        func_to_analyze = None
        # run() has greater priority
        if analyzer.run_function is not None:
            func_to_analyze = analyzer.run_function
        elif analyzer.train_and_predict is not None:
            func_to_analyze = analyzer.train_and_predict

        if func_to_analyze is not None:
            # Extract function signature information
            args_info = []
            
            for arg in func_to_analyze.args.args:
                arg_name = arg.arg
                arg_type = None
                args_info.append((arg_name, arg_type))
            
            return_type = None

            run_function_info = {
                'name': func_to_analyze.name,
                'args': args_info,
                'return_type': return_type
            }
        
        return has_entrypoint, run_function_info, None
    
    except SyntaxError as e:
        #raise e
        print()
        print(str(e))
        print()
        return True, None, f"{e=}"


def generate_entrypoint_code(run_function_info):
    """
    Generate the entrypoint code based on run function signature.
    
    Args:
        run_function_info: dict with function signature information
        
    Returns:
        str: Generated entrypoint code
    """
    if not run_function_info or not run_function_info['args']:
        return """
if __name__ == "__main__":
    res = run()
    print(res)
"""
    
    lines = ["\nif __name__ == \"__main__\":"]
    
    # Add pandas import if needed (detect if any args might be DataFrames)
    needs_pandas = any(arg_type and ('DataFrame' in str(arg_type) or 'pd.' in str(arg_type)) 
                      for _, arg_type in run_function_info['args'])
    
    if needs_pandas:
        lines.append("    import pandas as pd")
    
    # Generate CSV loading for each argument
    arg_names = []
    for arg_name, arg_type in run_function_info['args']:
        lines.append(f"    {arg_name} = pd.read_csv(\"{arg_name}.csv\")")
        arg_names.append(arg_name)
    
    # Generate function call
    args_str = ", ".join(arg_names)
    lines.append(f"    res = run({args_str})")
    lines.append("    print(res)")
    
    return "\n".join(lines) + "\n"


def generate_error_string(error) -> str:
    if error is None:
        return "\"\"\"CODE_PARSE_SUCCESS\n\"\"\"\n"
    return '\n'.join(["\"\"\"FATAL_ERROR", error.replace('"', '\\"'), "\"\"\"\n"])


def process_submission_files(input_dir, output_dir, add_entrypoint=False):
    """
    Recursively find all submission.py files, add path comments, and copy to output directory.
    
    Args:
        input_dir (str): Directory to search recursively for submission.py files
        output_dir (str): Directory where processed files will be written
        add_entrypoint (bool): Whether to add entrypoint if missing
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all submission.py files recursively
    submission_files = list(input_path.rglob("submission.py"))
    
    if not submission_files:
        print(f"No submission.py files found in {input_dir}")
        return
    
    print(f"Found {len(submission_files)} submission.py file(s)")
    
    for file_path in submission_files:
        try:
            # Read the original file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Process entrypoint if requested
            processed_content = original_content

            if add_entrypoint:
                has_entrypoint, run_function_info, error_code = analyze_entrypoint(original_content)
                if not has_entrypoint and run_function_info:
                    entrypoint_code = generate_entrypoint_code(run_function_info)
                    processed_content = original_content + entrypoint_code
                    print(f"  Added entrypoint to {file_path}")
                elif not has_entrypoint:
                    print(f"  No entrypoint found and no run() function in {file_path}")
           
            # Create the comment section
            path_comment = f"# {file_path}\n# " + (("ENTRYPOINT_ADDED" if (not has_entrypoint and run_function_info) else ("ENTRYPOINT_NOT_ADDED" if has_entrypoint else "ENTRYPOINT_ERROR_NO_FUNC")) if add_entrypoint else "ENTRYPOINT_NOT_CHECKED") + "\n"

            # Combine comment with processed content
            new_content = path_comment + generate_error_string(error_code) + processed_content
            
            # Create a unique filename for the output
            # Use the relative path from input_dir to maintain directory structure
            rel_path = file_path.relative_to(input_path)
            
            # Replace directory separators with underscores to flatten the structure
            # Keep the .py extension
            flat_name = str(rel_path).replace(os.sep, '_')
            if flat_name == "submission.py":
                # If it's directly in the root, use a simple name
                output_filename = "submission.py"
            else:
                # Remove the final _submission.py and replace with _submission.py
                output_filename = flat_name
            
            output_file_path = output_path / output_filename
            
            # Handle name conflicts by adding numbers
            counter = 1
            base_name = output_file_path.stem
            extension = output_file_path.suffix
            while output_file_path.exists():
                output_filename = f"{base_name}_{counter}{extension}"
                output_file_path = output_path / output_filename
                counter += 1
            
            # Write the new file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"Processed: {file_path} -> {output_file_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            raise e


def main():
    parser = argparse.ArgumentParser(
        description="Recursively find submission.py files, add path comments, and copy to output directory"
    )
    parser.add_argument(
        "input_dir",
        help="Input directory to search recursively for submission.py files"
    )
    parser.add_argument(
        "output_dir", 
        help="Output directory where processed files will be written"
    )
    parser.add_argument(
        "--add-entrypoint",
        action="store_true",
        help="Add entrypoint code if missing (looks for run() function)"
    )
    parser.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Preserve directory structure in output (creates subdirectories)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist or is not a directory")
        return 1
    
    if args.preserve_structure:
        process_with_structure(args.input_dir, args.output_dir, args.add_entrypoint)
    else:
        process_submission_files(args.input_dir, args.output_dir, args.add_entrypoint)
    
    return 0


def process_with_structure(input_dir, output_dir, add_entrypoint=False):
    """
    Alternative processing that preserves directory structure.
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    
    # Find all submission.py files recursively
    submission_files = list(input_path.rglob("submission.py"))
    
    if not submission_files:
        print(f"No submission.py files found in {input_dir}")
        return
    
    print(f"Found {len(submission_files)} submission.py file(s)")
    
    for file_path in submission_files:
        try:
            # Read the original file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Process entrypoint if requested
            processed_content = original_content
            if add_entrypoint:
                has_entrypoint, run_function_info, error_code = analyze_entrypoint(original_content)
                if not has_entrypoint and run_function_info:
                    entrypoint_code = generate_entrypoint_code(run_function_info)
                    processed_content = original_content + entrypoint_code
                    print(f"  Added entrypoint to {file_path}")
                elif not has_entrypoint:
                    print(f"  No entrypoint found and no run() function in {file_path}")
           
            # Create the comment section
            path_comment = f"# {file_path}\n# " + (("ENTRYPOINT_ADDED" if (not has_entrypoint and run_function_info) else ("ENTRYPOINT_NOT_ADDED" if has_entrypoint else "ENTRYPOINT_ERROR_NO_FUNC")) if add_entrypoint else "ENTRYPOINT_NOT_CHECKED") + "\n"

            # Combine comment with processed content
            new_content = path_comment + generate_error_string(error_code) + processed_content
            
            # Preserve directory structure
            rel_path = file_path.relative_to(input_path)
            output_file_path = output_path / rel_path
            
            # Create parent directories if they don't exist
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the new file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"Processed: {file_path} -> {output_file_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            raise e


if __name__ == "__main__":
    exit(main())
