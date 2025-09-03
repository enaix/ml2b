#!/bin/bash
set -e

# Usage
usage() {
    echo "Usage: $0 [-r|--rebuild] submission_script competition_id bench_mode [folds]"
    echo ""
    echo "  -r, --rebuild      Rebuild the container before running"
    echo ""
    echo "Arguments:"
    echo "  submission_script  Path to the submission script"
    echo "  competition_id     Competition identifier"
    echo "  bench_mode         Benchmarking mode (MONO_PREDICT or MODULAR_PREDICT)"
    echo ""
    echo "Optional:"
    echo "  folds              Override the number of folds to run"
    exit 1
}

# Store positional arguments
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--rebuild) REBUILD=true ;;
        *) POSITIONAL+=("$1") ;;
    esac
    shift
done
set -- "${POSITIONAL[@]}"

# Assign positional arguments to variables
SUBMISSION_SCRIPT="${POSITIONAL[0]}"
COMPETITION_ID="${POSITIONAL[1]}"
BENCH_MODE="${POSITIONAL[2]}"

if [[ ${#POSITIONAL[@]} -gt 3 ]]; then
	BENCH_FOLDS_OVERRIDE="${POSITIONAL[3]}"
fi

# Check for required arguments
if [[ -z $SUBMISSION_SCRIPT || -z $COMPETITION_ID || -z $BENCH_MODE ]]; then
    usage
fi

# Prepare the folders
if [[ -d "./python/submission" ]]; then
    rm -r ./python/submission
fi
mkdir -p ./python/submission/
mkdir -p ./python/data
echo "" > ./python/submission/__init__.py

# Copy the submission script
cp "$SUBMISSION_SCRIPT" ./python/submission/code.py

# Competition init
export COMPETITION_ID="$COMPETITION_ID"
export BENCH_LANG="English"
export BENCH_MODE="$BENCH_MODE"
export BENCH_FOLDS_OVERRIDE="$BENCH_FOLDS_OVERRIDE"

# Rebuild container if requested
if [[ "$REBUILD" == "true" ]]; then
    docker compose build bench_python
fi

# Execute
docker compose run bench_python

cat ./python/submission/results.json
