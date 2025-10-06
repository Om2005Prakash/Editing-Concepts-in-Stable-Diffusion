#!/bin/bash

# List of concepts (flat array: concept1 concept2 concept1 concept2 ...)
concepts=(
  "LabradorRetriever" "Cat"
  "BarbetonDaisy" "Hibiscus"
  "BlueJay" "Parrot"
  "GolfBall" "TennisBall"
  "AppleFruit" "OrangeFruit"
  "VanGough" "Monet"
  "Doodle" "Realistic"
  "Neon" "Countryside"
  "Monet" "VanGogh"
  "Sketch" "Realistic"
  "Wedding" "FamilyReunion"
  "Sunset" "Afternoon"
  "Rainfall" "Sunny"
  "AuroraBorialis" "Afternoon"
  "Scenery" "Interior"
  "Sleeping" "Sitting"
  "Walking" "Sitting"
  "Eating" "Walking"
  "Dancing" "Sitting"
  "Jumping" "Sitting"
)

num_gpus=4

# Control how many jobs run in parallel batches
parallel=1  # 1 job per GPU
batch_size=$((parallel * num_gpus))

i=0
total=${#concepts[@]}

while [ $i -lt $total ]; do
  echo "Starting batch with concepts index $i to $((i+batch_size*2-1))..."

  for ((j=0; j<batch_size && i<total; j++, i+=2)); do
    gpu_id=$((j % num_gpus))
    concept1="${concepts[i]}"
    concept2="${concepts[i+1]}"
    echo "Launching '$concept1' vs '$concept2' on GPU $gpu_id"
    python edit_script.py "$concept1" "$concept2" "cuda:$gpu_id" &
  done

  # Wait for this batch to finish before starting the next
  wait
  echo "Batch completed."
done

echo "All runs completed."