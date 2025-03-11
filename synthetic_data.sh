#!/bin/bash


folders=("Brownian" "Sines_se" "VdP")
cd dataset

for folder in "${folders[@]}"; do
    lower_name=$(echo "$folder" | tr '[:upper:]' '[:lower:]')
    script_name="make_${lower_name}_data.py" 

    if [[ -d "$folder" && -f "$folder/$script_name" ]]; then
        echo "Running $script_name in $folder..."
        (cd "$folder" && python "$script_name")
    fi
    
done
