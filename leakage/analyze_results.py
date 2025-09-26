import os
import re
import sys
import ast
import argparse
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List



# Functions analysis code
def find_funcs_for_loc(fname: os.PathLike, loc: List) -> List:
    """
    Find Python functions that contain the specified line numbers.

    Args:
        fname: Path to the Python file
        loc: List of line numbers to search for

    Returns:
        List of dictionaries containing function information:
        - 'name': Function name
        - 'args': List of argument names
        - 'start_line': Starting line number of the function
        - 'end_line': Ending line number of the function
        - 'contains_lines': List of input line numbers that fall within this function
    """
    if not loc:
        return []

    # Read the file content
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines()
    except (IOError, OSError) as e:
        raise FileNotFoundError(f"Could not read file {fname}: {e}")

    # Parse the AST
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError(f"Could not parse Python file {fname}: {e}")

    # Find all function definitions
    functions = []

    class FunctionVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            # Get function arguments
            args = []

            # Regular arguments
            for arg in node.args.args:
                args.append(arg.arg)

            # *args
            if node.args.vararg:
                args.append(f"*{node.args.vararg.arg}")

            # Keyword-only arguments
            for arg in node.args.kwonlyargs:
                args.append(arg.arg)

            # **kwargs
            if node.args.kwarg:
                args.append(f"**{node.args.kwarg.arg}")

            # Find the end line of the function
            end_line = node.lineno

            # Walk through all nodes in the function to find the maximum line number
            for child in ast.walk(node):
                if hasattr(child, 'lineno') and child.lineno > end_line:
                    end_line = child.lineno

            # Also check the last line with content in the function's scope
            # by looking at indentation
            start_line = node.lineno
            base_indent = None

            # Find base indentation of the function
            if start_line <= len(lines):
                func_line = lines[start_line - 1]
                base_indent = len(func_line) - len(func_line.lstrip())

            # Scan forward from the AST end_line to find actual end
            if base_indent is not None:
                for i in range(end_line, len(lines)):
                    line = lines[i]
                    if line.strip():  # Non-empty line
                        line_indent = len(line) - len(line.lstrip())
                        if line_indent <= base_indent and not line.lstrip().startswith(('"""', "'''")):
                            # Found a line at same or lower indentation, function ends at previous line
                            break
                    end_line = i + 1

            functions.append({
                'name': node.name,
                'args': args,
                'start_line': start_line,
                'end_line': end_line,
                'node': node  # Keep for nested function handling
            })

            # Continue visiting for nested functions
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            # Handle async functions the same way
            self.visit_FunctionDef(node)

    visitor = FunctionVisitor()
    visitor.visit(tree)

    # Remove the 'node' key as it's not needed in the output
    for func in functions:
        del func['node']

    # Filter functions that contain any of the specified line numbers
    result = []
    for func in functions:
        contains_lines = []
        for line_num in loc:
            if func['start_line'] <= line_num <= func['end_line']:
                contains_lines.append(line_num)

        if contains_lines:
            func['contains_lines'] = contains_lines
            result.append(func)

    return list({v["start_line"]:v for v in result}.values())  # find unique [start, end]



def extract_python_file_info(py_file_path):
    """
    Extract information from Python file.
    
    Returns:
        dict: Contains file_path, entrypoint_status, error_status, and traceback
    """
    with open(py_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Extract path from first line (remove leading #)
    original_path = lines[0].strip().lstrip('#').strip() if lines else ""
    
    # Extract entrypoint status from second line (remove leading #)
    entrypoint_status = lines[1].strip().lstrip('#').strip() if len(lines) > 1 else ""

    traceback = None
  
    if len(lines) >= 4:
        # Parse the error string
        error_status = lines[2][3:]   # Remove the """ comment start
        if lines[3] != "\"\"\"":
            traceback = ""
            for i in range(3, len(lines)):
                if lines[i] != "\"\"\"":
                    traceback += lines[i] + " "
                else:
                    break
    else:
        error_status = "BAD_HEADER_FORMAT"
        traceback = None
    return {
        'original_file_path': original_path,
        'entrypoint_status': entrypoint_status,
        'error_status': error_status,
        'traceback': traceback
    }


def extract_html_info(html_file_path):
    """
    Extract leakage counts, highlight lines, and mark_leak_lines from HTML file.
    
    Returns:
        dict: Contains leakage counts, highlight lines, and mark leak lines
    """
    with open(html_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    
    # Initialize results
    preprocessing_leakage = 0
    overlap_leakage = 0
    no_independence = 0
    highlight_lines = []
    mark_leak_lines = []
    warnings        = []
    
    # Extract leakage counts from the summary table
    sum_table = soup.find('table', class_='sum')
    if sum_table:
        rows = sum_table.find_all('tr')
        for row in rows[1:]:  # Skip header row
            cells = row.find_all('td')
            if len(cells) >= 2:
                leakage_type = cells[0].get_text(strip=True)
                count = cells[1].get_text(strip=True)
                
                try:
                    count_int = int(count)
                    if 'Pre-processing leakage' in leakage_type:
                        preprocessing_leakage = count_int
                    elif 'Overlap leakage' in leakage_type:
                        overlap_leakage = count_int
                    elif 'No independence test data' in leakage_type:
                        no_independence = count_int
                except ValueError:
                    pass

    all_buttons = soup.find_all('button')

    for button in all_buttons:
        # Find all red buttons (aka warnings)
        style = button.get('style', '')
        is_match = re.search(r'background-color: red', style)
        if is_match:
            warnings.append(button.text)

    # Extract highlight lines from buttons
    highlight_buttons = soup.find_all('button', string=re.compile(r'highlight train/test sites'))
    
    for button in highlight_buttons:
        onclick_attr = button.get('onclick', '')
        # Extract lines from onclick="highlight_lines([A, B, C, ...])"
        match = re.search(r'highlight_lines\(\[(.+?)\]\)', onclick_attr)
        if match:
            lines_str = match.group(1)
            # Parse the line numbers
            line_numbers = re.findall(r'(\d+)', lines_str)
            if line_numbers:
                highlight_lines.extend([int(line) for line in line_numbers])
    
    # Extract mark leak lines from buttons
    mark_buttons = soup.find_all('button', string=re.compile(r'show and go to first leak src'))
    
    for button in mark_buttons:
        onclick_attr = button.get('onclick', '')
        # Extract lines from onclick="mark_leak_lines([A, B, C, ...])"
        match = re.search(r'mark_leak_lines\(\[(.+?)\]\)', onclick_attr)
        if match:
            lines_str = match.group(1)
            # Parse the line numbers
            line_numbers = re.findall(r'(\d+)', lines_str)
            if line_numbers:
                mark_leak_lines.extend([int(line) for line in line_numbers])
    
    return {
        'preprocessing_leakage': preprocessing_leakage,
        'overlap_leakage': overlap_leakage,
        'no_independence': no_independence,
        'highlight_lines': highlight_lines,
        'mark_leak_lines': mark_leak_lines,
        'warnings': warnings
    }


def process_files_in_folder(folder_path):
    """
    Process all Python files and their corresponding HTML files in a folder.
    Skip files with .ir.py extension.
    
    Args:
        folder_path (str): Path to the folder containing the files
        
    Returns:
        pd.DataFrame: DataFrame with all extracted information
    """
    folder_path = Path(folder_path)
    current_dir = Path.cwd()
    
    data = []
    
    # Find all Python files, excluding .ir.py files
    python_files = [f for f in folder_path.glob('*.py') if not f.name.endswith('.ir.py')]
    
    for py_file in python_files:
        # Get corresponding HTML file
        html_file = py_file.with_suffix('.html')
        
        # Extract info from Python file
        py_info = extract_python_file_info(py_file)
        
        # Get relative paths
        py_relative_path = os.path.relpath(py_file, current_dir)
        
        # Initialize HTML info with NA values
        html_info = {
            'preprocessing_leakage': pd.NA,
            'overlap_leakage': pd.NA,
            'no_independence': pd.NA,
            'highlight_lines': pd.NA,
            'mark_leak_lines': pd.NA,
            'warnings': pd.NA
        }
        html_relative_path = pd.NA
        
        # Extract info from HTML file if it exists
        if html_file.exists():
            html_info = extract_html_info(html_file)
            html_relative_path = os.path.relpath(html_file, current_dir)
        else:
            print(f"Warning: No corresponding HTML file found for {py_file}")

        # Extract LOC <-> function info

        if html_info["mark_leak_lines"] is not pd.NA:
           funcs_info = find_funcs_for_loc(py_file, html_info["mark_leak_lines"])
        else:
            funcs_info = pd.NA

        # Combine all information
        row_data = {
            'python_file_path': py_relative_path,
            'html_file_path': html_relative_path,
            **py_info,
            **html_info,
            'funcs_info': funcs_info
        }
        
        data.append(row_data)
    
    return pd.DataFrame(data)


def main():
    """
    Main function to process files and save results.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process Python and HTML file analysis results')
    parser.add_argument('input_folder', help='Folder containing Python and HTML files')
    parser.add_argument('-o', '--output', default='file_analysis_results.csv', 
                       help='Output CSV file path (default: file_analysis_results.csv)')
    
    args = parser.parse_args()
    
    folder_path = args.input_folder
    output_file = args.output
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)
    
    # Process files
    df = process_files_in_folder(folder_path)
    
    if df.empty:
        print("No Python files found in the specified folder.")
        return
    
    # Display results
    print(f"\nProcessed {len(df)} Python files.")
    print(f"Found {df['html_file_path'].notna().sum()} corresponding HTML files.")
    print("\nDataFrame columns:")
    print(df.columns.tolist())
    
    # Show first few rows
    print(f"\nFirst {min(3, len(df))} rows:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(df.head(3))
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to '{output_file}'")
    
    return df


if __name__ == "__main__":
    df = main()
