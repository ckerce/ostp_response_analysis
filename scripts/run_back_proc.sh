#!/bin/bash

# Define the overall range and chunk size parameters
overall_start=10062
overall_end=9062 # The loop will stop *before* this if not perfectly divisible
chunk_size=100
step=-$chunk_size

echo "Starting processing..."
echo "Overall range: $overall_start down towards $overall_end"
echo "Chunk size: $chunk_size"
echo "---"

# Loop through the starting point of each chunk
# The seq command generates the STARTING number for each chunk (10062, 9962, 9862, ..., 9162)
for chunk_start in $(seq $overall_start $step $overall_end); do

  # Calculate the actual last number IN the chunk (inclusive)
  # Example: If chunk_start is 10062, chunk_end is 10062 - 100 + 1 = 9963
  chunk_end=$((chunk_start - chunk_size + 1))

  # Optional: Ensure chunk_end doesn't go below the overall desired minimum
  # if [[ $chunk_end -lt $overall_end ]]; then
  #   chunk_end=$overall_end
  # fi
  # (In your specific case 10062 -> 9062 step -100, the last chunk starts at 9162
  # and its calculated end is 9063, which is >= 9062, so this check isn't strictly needed here)

  echo "Processing chunk: $chunk_start down to $chunk_end"

  # Generate the sequence for this specific chunk, space-separated
  # Example: seq -s ' ' 10062 -1 9963 generates "10062 10061 ... 9963"
  numbers_in_chunk=$(seq -s ' ' $chunk_start -1 $chunk_end)

  # Define the output filename based on the chunk's end value (as in your original code's apparent intent)
  # Example: v2_gemma3-27b-out_9963.md
  output_file="v2_gemma3-27b-out_${chunk_end}.md"

  echo "Output file: $output_file"
  echo "Running python script..."

  # Execute the python script, passing the generated numbers as arguments
  # WARNING: If chunk_size is very large, this might exceed the command-line argument length limit.
  # Consider modifying the python script to read from stdin or a file if that happens.
  python v2_analyze_rfi.py $numbers_in_chunk > "$output_file"

  echo "Finished chunk processing."
  echo "---"

done

echo "All chunks processed."
