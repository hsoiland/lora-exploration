#!/bin/bash
# Run training with output logged to a file for remote monitoring,
# but display a progress bar for better user experience

# Check if script name was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <script_name> [args...]"
    echo "Example: $0 train_full_lora.py --output_dir gina_full_lora"
    exit 1
fi

SCRIPT=$1
shift  # Remove script name from args
ARGS="$@"  # All remaining arguments

# Create log directory in output dir
OUTPUT_DIR=$(echo "$ARGS" | grep -oP -- "--output_dir \K[^ ]+")
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="training_output"  # Default if not specified
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/train.log"

echo "Starting training with: python $SCRIPT $ARGS"
echo "Output will be logged to: $LOG_FILE"
echo "Use 'tail -f $LOG_FILE' from another terminal to monitor detailed progress"
echo ""

# Run with progress bar visible but redirect full output to log file
PYTHONIOENCODING=utf-8 python -u "$SCRIPT" $ARGS 2>&1 | tee "$LOG_FILE" | grep -E "Epoch|Loss:|100%|LoRA|saving|✅|⚠️|❌"

echo ""
echo "Training complete. Log saved to $LOG_FILE" 