#!/bin/bash

BASE="https://docs.google.com/spreadsheets/d/1ZY8NRI-WZ4RoDK8GpEy_GTSSWVZPxQQthTaySp5jnao/export?format=csv&gid="

declare -A sheets=(
    ["1525338984"]="Arab.csv"
    ["1607321930"]="Belarus.csv"
    ["940745352"]="Chinese.csv"
    ["658946950"]="English.csv"
    ["262900029"]="Italian.csv"
    ["1957726072"]="Japanese.csv"
    ["388903984"]="Kazakh.csv"
    ["2122623213"]="Polish.csv"
    ["1490673460"]="Romanian.csv"
    ["1065613449"]="Spanish.csv"
    ["0"]="Turkish.csv"
    ["751010443"]="Russian.csv"
    ["11714048"]="French.csv"
)

DEST="competitions/tasks"
mkdir -p "$DEST"
for gid in "${!sheets[@]}"; do
    curl -L "${BASE}${gid}" -o "${DEST}/${sheets[$gid]}"
done