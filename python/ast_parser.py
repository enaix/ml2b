import ast
import importlib.util
import types
from typing import Optional, Any
import os


class TopLevelExecutionRemover(ast.NodeTransformer):
    """
    AST transformer that removes top-level executable statements while preserving
    function definitions, class definitions, and import statements.
    """

    def __init__(self):
        self.preserved_node_types = {
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Import,
            ast.ImportFrom,
            ast.If,  # We'll handle if __name__ == "__main__" specially
        }
        # Track global variable assignments that might be needed
        self.global_assignments = set()

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Process the module, filtering out unwanted top-level statements."""
        new_body = []

        for stmt in node.body:
            if self._should_preserve_statement(stmt):
                new_body.append(self.visit(stmt))
            elif isinstance(stmt, ast.Assign):
                # Preserve assignments to variables that look like constants or configurations
                if self._is_likely_constant_assignment(stmt):
                    new_body.append(self.visit(stmt))

        return ast.Module(body=new_body, type_ignores=node.type_ignores)

    def visit_If(self, node: ast.If) -> Optional[ast.If]:
        """Handle if statements, preserving if __name__ == "__main__" blocks."""
        # if self._is_main_guard(node):
        #     # Skip the entire if __name__ == "__main__" block
        #     return None
        # Keep other if statements (they might be configuration logic)
        return self.generic_visit(node)

    def _should_preserve_statement(self, stmt: ast.stmt) -> bool:
        """Determine if a statement should be preserved."""
        return type(stmt) in self.preserved_node_types

    def _is_likely_constant_assignment(self, stmt: ast.Assign) -> bool:
        """
        Check if an assignment looks like a constant or configuration variable.
        Preserves assignments like: DEBUG = True, API_KEY = "...", etc.
        """
        if len(stmt.targets) != 1:
            return False

        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            return False

        name = target.id
        # Preserve assignments that look like constants (UPPER_CASE)
        # or common configuration patterns
        if (name.isupper() or
            name.startswith('_') or
            any(pattern in name.lower() for pattern in ['config', 'setting', 'default'])):
            return True

        # Preserve simple literal assignments
        if isinstance(stmt.value, (ast.Constant, ast.Num, ast.Str, ast.NameConstant)):
            return True

        return False

    def _is_main_guard(self, node: ast.If) -> bool:
        """Check if this is an if __name__ == "__main__" statement."""
        if not isinstance(node.test, ast.Compare):
            return False

        compare = node.test
        if (isinstance(compare.left, ast.Name) and
            compare.left.id == '__name__' and
            len(compare.ops) == 1 and
            isinstance(compare.ops[0], ast.Eq) and
            len(compare.comparators) == 1):

            comparator = compare.comparators[0]
            if isinstance(comparator, ast.Constant):
                return comparator.value == "__main__"
            elif isinstance(comparator, ast.Str):  # Python < 3.8
                return comparator.s == "__main__"

        return False


def safe_import_from_file(file_path: str, module_name: Optional[str] = None) -> types.ModuleType:
    """
    Safely import a Python file by removing top-level executable code.

    Args:
        file_path: Path to the Python file to import
        module_name: Optional name for the module (defaults to filename without extension)

    Returns:
        The imported module with only definitions and safe assignments

    Raises:
        FileNotFoundError: If the file doesn't exist
        SyntaxError: If the file contains invalid Python syntax
        ImportError: If there are issues during import
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if module_name is None:
        module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Read the source code
    with open(file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()

    # Parse the AST
    try:
        tree = ast.parse(source_code, filename=file_path)
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in {file_path}: {e}")

    # Transform the AST to remove unwanted statements
    transformer = TopLevelExecutionRemover()
    filtered_tree = transformer.visit(tree)

    # Fix missing locations for the modified AST
    ast.fix_missing_locations(filtered_tree)

    # Compile the modified AST
    try:
        compiled_code = compile(filtered_tree, file_path, 'exec')
    except Exception as e:
        raise ImportError(f"Failed to compile filtered code from {file_path}: {e}")

    # Create a new module
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)

    # Execute the filtered code in the module's namespace
    try:
        exec(compiled_code, module.__dict__)
    except Exception as e:
        raise ImportError(f"Failed to execute filtered code from {file_path}: {e}")

    # Add the module to sys.modules for proper import behavior
    # sys.modules[module_name] = module

    return module


def get_functions_from_module(module: types.ModuleType) -> dict[str, Any]:
    """
    Extract all functions from a module.

    Args:
        module: The imported module

    Returns:
        Dictionary mapping function names to function objects
    """
    functions = {}
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and not name.startswith('_'):
            functions[name] = obj
    return functions


