# grade_manual.py
import os
import sys
import subprocess
import argparse

from src.bench import BenchPipeline

def main():
    parser = argparse.ArgumentParser(description='Prepare data and run a manual benchmark test using Docker')
    parser.add_argument('competition_id', help='ID of the competition to run')
    parser.add_argument('lang', help='Competition language')
    parser.add_argument('code_path', help='Path to the Python code file to test')
    parser.add_argument('--mode', '-m', choices=['mono', 'modular'], default='mono',
                       help='Execution mode: mono (monolithic) or modular (default: mono)')
    parser.add_argument('--extended_schema', '-e', choices=['y', 'n'], default='n', help='Use extended schema for submission code')
    parser.add_argument('--folds', '-f', type=int, help='Override number of folds')
    parser.add_argument('--rebuild', '-r', action='store_true', 
                       help='Rebuild Docker container before running')
    parser.add_argument('--seed', '-s', type=int, default=42)

    args = parser.parse_args()

    print(f"=== Manual Benchmark Test: {args.competition_id} ===")

    # 1. PREPARE THE DATA on the host using the modern pipeline
    print("[1/3] Preparing data on host using BenchPipeline...")
    pipeline = BenchPipeline(basepath=os.getcwd(), prepare_data=True)
    
    target_comp = None
    for comp in pipeline.competitions:
        if comp.comp_id == args.competition_id:
            target_comp = comp
            break
            
    if target_comp is None:
        print(f"Error: Competition '{args.competition_id}' not found.")
        sys.exit(1)
    
    pipeline.prepare_train_data(target_comp, args.seed)
    print("✓ Data preparation complete on host.")

    # 1.5 Ensure submission package __init__.py exists for Docker grading
    submission_dir = os.path.join(
        'python', 'submission', f"submission_{args.competition_id}-{args.lang}-python-only_code"
    )
    os.makedirs(submission_dir, exist_ok=True)
    init_file = os.path.join(submission_dir, '__init__.py')
    try:
        if not os.path.exists(init_file):
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write('')
        print(f"✓ Ensured package file at {init_file}")
    except Exception as e:
        print(f"Warning: Could not create __init__.py at {init_file}: {e}")

    # 2. Call the existing grade.sh script to handle Docker
    print("[2/3] Invoking grade.sh to run inside Docker container...")
    try:
        cmd = [
            './grade.sh',
            args.code_path,
            args.competition_id,
            args.lang,
            'MONO_PREDICT' if args.mode == 'mono' else 'MODULAR_PREDICT',
            args.extended_schema
        ]
        
        if args.rebuild:
            cmd.insert(1, '--rebuild')
        if args.folds:
            cmd.append(str(args.folds))

        # This runs the grade.sh script, which builds the container and runs bench.py inside it
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print("Error running grade.sh:")
        print(e.stderr)
        print(f"Grade.sh failed with return code {e.returncode}")
    except Exception as e:
        print(f"Error: {e}")

    # 3. CLEANUP
    print("[3/3] Cleaning up prepared data on host...")
    pipeline.erase_train_data(target_comp)
    print("✓ Cleanup complete.")
    print("=== Manual test finished ===")

if __name__ == "__main__":
    main()
