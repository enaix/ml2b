#!/bin/bash
set -e

# Usage
usage() {
    echo "Usage: $0 [-r|--rebuild] submission_script competition_id lang bench_mode extended_schema [folds]"
    echo ""
    echo "  -r, --rebuild      Rebuild the container before running"
    echo ""
    echo "Arguments:"
    echo "  submission_script  Path to the submission script"
    echo "  competition_id     Competition identifier"
    echo "  lang               Competition language"
    echo "  bench_mode         Benchmarking mode (MONO_PREDICT or MODULAR_PREDICT)"
    echo "  extended_schema    Use extended schema for submission code"
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
BENCH_LANG="${POSITIONAL[2]}"
BENCH_MODE="${POSITIONAL[3]}"
EXTENDED_SCHEMA="${POSITIONAL[4]}"

if [[ ${#POSITIONAL[@]} -gt 5 ]]; then
	BENCH_FOLDS_OVERRIDE="${POSITIONAL[5]}"
fi

# Check for required arguments
if [[ -z $SUBMISSION_SCRIPT || -z $COMPETITION_ID || -z $BENCH_LANG || -z $BENCH_MODE || -z $EXTENDED_SCHEMA ]]; then
    usage
fi

# Calculate the submission name
SUBMISSION_NAME="submission_${COMPETITION_ID}-${BENCH_LANG}-python-only_code"

# Prepare the folders
if [[ ! -d "./python/submission" ]]; then
    mkdir -p ./python/submission/
fi

if [[ -d "./python/submission/${SUBMISSION_NAME}" ]]; then
    echo "Directory exists: ./python/submission/${SUBMISSION_NAME}. Please move or delete old competition files"
    exit 1
fi


echo "" > "./python/submission/$SUBMISSION_NAME/__init__.py"

# Copy the submission script
cp "$SUBMISSION_SCRIPT" "./python/submission/$SUBMISSION_NAME/code.py"

# Competition init
export COMPETITION_ID="$COMPETITION_ID"
export SUBMISSION_NAME="$SUMBISSION_NAME"
export BENCH_LANG="$BENCH_LANG"
export BENCH_MODE="$BENCH_MODE"
export BENCH_FOLDS_OVERRIDE="$BENCH_FOLDS_OVERRIDE"
export EXTENDED_SCHEMA="$EXTENDED_SCHEMA"

# Rebuild container if requested
if [[ "$REBUILD" == "true" ]]; then
    docker compose build bench_python
fi

# Mount the submission folder using the override
cat > docker-compose.override.yml << EOF
services:
  bench_python:
    volumes:
      - ./python/submission/${SUBMISSION_NAME}/:/home/bench/submission
EOF


# Execute
docker compose run bench_python

# Remove the override
rm docker-compose.override.yml


echo "\nResults for ./python/submission/${SUBMISSION_NAME}"
echo "======================"
cat ./python/submission/${SUBMISSION_NAME}/results.json

