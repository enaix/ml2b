#!/bin/bash
set -e

if [[ $# -lt 2 ]]; then
    echo "Usage: make_subfolders.sh BATCHES LARGE_FOLDER"
    echo "Puts files in the LARGE_FOLDER into multiple subfolders. Remainder is added to the N+1-th subfolder"
    exit 1
fi

# https://stackoverflow.com/a/29118145
BATCHES="$1"
LARGE_FOLDER="$2"
FILES_TOTAL=$(find "$LARGE_FOLDER" -maxdepth 1 -type f | wc -l)
FILES_PER_BATCH=$(($FILES_TOTAL / $BATCHES))
#REMAINING=$(($FILES_TOTAL % $BATCHES))

for i in `seq 1 $(($BATCHES+1))`;
do
    mkdir -p "$LARGE_FOLDER/subfolder$i";
    find "$LARGE_FOLDER" -maxdepth 1 -type f | head -n $FILES_PER_BATCH | xargs -i mv "{}" "$LARGE_FOLDER/subfolder$i"
done
