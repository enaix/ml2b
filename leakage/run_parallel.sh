#!/bin/bash
set -e

if [[ $# -ne 2 ]]; then
    echo "Usage: run_parallel.sh COMMAND TARGET"
    echo "Executes the command for N subfolders in the TARGET directory. Subdirectory is appended to the end of the command"
    exit 1
fi

COMMAND="$1"
TARGET="$2"

for i in $(find $TARGET -type d); do
    $COMMAND "$i" 2>&1 | tee "$i.log" &
done

wait
