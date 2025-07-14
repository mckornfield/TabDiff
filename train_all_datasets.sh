#!/bin/bash

# Array of dataset names
datasets=("default" "shoppers" "magic" "beijing" "news" "diabetes")

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    echo "=========================================="
    echo "Training on dataset: $dataset"
    echo "=========================================="
    
    # Run the training command
    python main.py --dataname "$dataset" --mode train
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✅ Training completed successfully for $dataset"
    else
        echo "❌ Training failed for $dataset"
        echo "Continuing with next dataset..."
    fi
    
    echo ""
done

echo "=========================================="
echo "All training jobs completed!"
echo "==========================================" 