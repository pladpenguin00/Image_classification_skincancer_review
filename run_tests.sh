#!/bin/bash

# Define the different sets of arguments you want to test
declare -a models=("resnet18" "resnet34" "resnet50")
declare -a losses=("BCELoss" "CrossEntropyLoss" "MSELoss" "NLLLoss" "SmoothL1Loss")
declare -a optimizers=("Adam" "SGD" "RMSprop" "Adagrad" "AdamW")

# Iterate over each combination of model, loss, and optimizer
for model in "${models[@]}"; do
  for loss in "${losses[@]}"; do
    for optimizer in "${optimizers[@]}"; do
      echo "Running test with model=$model, loss=$loss, optimizer=$optimizer"
      python main.py --model "$model" --loss "$loss" --optimizer "$optimizer"
    done
  done
done