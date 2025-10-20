#!/bin/bash

# Script to run multiple rounds of experiments with launch_batch.py
# Usage: ./run_n_rounds.sh <config_file> <number_of_rounds>

# Load Python module on the cluster
if command -v ml &> /dev/null; then
    ml python/3.12.1
fi

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <config_file> <number_of_rounds>"
    echo "Example: $0 configs/experiment_config.json 5"
    exit 1
fi

CONFIG_FILE="$1"
NUM_ROUNDS="$2"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Check if number of rounds is a positive integer
if ! [[ "$NUM_ROUNDS" =~ ^[0-9]+$ ]] || [ "$NUM_ROUNDS" -eq 0 ]; then
    echo "Error: Number of rounds must be a positive integer"
    exit 1
fi

# Read the original RUN_NAME from the config file
ORIGINAL_RUN_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['RUN_NAME'])")

echo "Starting $NUM_ROUNDS rounds of experiments"
echo "Using config: $CONFIG_FILE"
echo "Base run name: $ORIGINAL_RUN_NAME"
echo "----------------------------------------"

echo "Ensuring typeguard is installed..."
python3 -m pip install typeguard --quiet

# Run the specified number of rounds
for i in $(seq 1 $NUM_ROUNDS); do
    echo ""
    echo "Starting round $i of $NUM_ROUNDS"
    
    NEW_RUN_NAME="${ORIGINAL_RUN_NAME}-${i}"
    
    echo "Run name for round $i: $NEW_RUN_NAME"
    echo "Launching batch job..."
    
    # Launch the batch job with CLI override for RUN_NAME
    python3 launch_batch.py --CONFIG "$CONFIG_FILE" --RUN_NAME "$NEW_RUN_NAME"
    
    # Check if launch was successful
    if [ $? -eq 0 ]; then
        echo "Round $i launched successfully"
    else
        echo "Error: Failed to launch round $i"
        exit 1
    fi
    
    # Small delay between rounds to avoid overwhelming the scheduler
    if [ $i -lt $NUM_ROUNDS ]; then
        echo "Waiting 5 seconds before next round..."
        sleep 5
    fi
done

echo ""
echo "----------------------------------------"
echo "All $NUM_ROUNDS rounds have been submitted!"
echo "Run names used:"
for i in $(seq 1 $NUM_ROUNDS); do
    echo "  - ${ORIGINAL_RUN_NAME}-${i}"
done
