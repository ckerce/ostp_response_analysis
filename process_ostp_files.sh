#!/bin/bash

# Strict mode: exit on error, exit on pipe fail, treat unset variables as an error
set -euo pipefail

# --- Configuration ---
# Number of file designators to process in each chunk
CHUNK_SIZE=100
# Command to execute the Python script.
# Ensure 'unified_rfi_analyzer.py' is executable and in PATH, or provide the full path.
PYTHON_EXECUTABLE="python"
PYTHON_SCRIPT_PATH="unified_rfi_analyzer.py"
# Output directory for results
OUTPUT_DIR="gemini-out"

# --- Helper Functions ---
log_message() {
    # Simple logger: prints timestamp and message to stdout
    # For more advanced logging, consider redirecting this to a log file as well.
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

print_usage() {
    echo "Usage: $0 <start_index_k> [num_chunks_N]"
    echo "  <start_index_k>: The 0-based starting index for the first chunk of files."
    echo "  [num_chunks_N]: Optional. The total number of ${CHUNK_SIZE}-item chunks to process."
    echo "Example (process files 0-99): $0 0"
    echo "Example (process files 0-99, then 100-199, then 200-299): $0 0 3"
}

# --- Argument Parsing and Validation ---
# Check if the first argument (start_index_k) is provided
if [[ -z "${1-}" ]]; then # Use ${1-} to avoid unset variable error if set -u is active and no arg
    log_message "Error: Starting index (k) not provided."
    print_usage
    exit 1
fi

start_index_k=$1
num_chunks_N=${2-} # Default to empty if not provided

# Validate start_index_k: must be a non-negative integer
if ! [[ "$start_index_k" =~ ^[0-9]+$ ]]; then
    log_message "Error: Starting index '$start_index_k' must be a non-negative integer."
    print_usage
    exit 1
fi

# Determine the number of iterations (chunks to process)
num_iterations=1 # Default to 1 chunk if num_chunks_N is not provided
if [[ -n "$num_chunks_N" ]]; then
    # Validate num_chunks_N: must be a positive integer if provided
    if ! [[ "$num_chunks_N" =~ ^[1-9][0-9]*$ ]]; then
        log_message "Error: Number of chunks '$num_chunks_N' must be a positive integer."
        print_usage
        exit 1
    fi
    num_iterations=$num_chunks_N
fi

# --- Directory Setup ---
# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
if [[ $? -ne 0 ]]; then
    log_message "Error: Could not create output directory '$OUTPUT_DIR'. Check permissions."
    exit 1
fi

# --- Main Processing Loop ---
overall_script_success=0 # 0 for overall success, 1 for any failure

# Loop for the specified number of chunks
for (( i=0; i<num_iterations; i++ )); do
    # Calculate the start and end indices for the current chunk
    # chunk_actual_start is the 0-based index for the Python script
    chunk_actual_start=$((start_index_k + i * CHUNK_SIZE))
    chunk_actual_end=$((chunk_actual_start + CHUNK_SIZE - 1))

    # Generate the sequence of indices for the Python script
    # These will be space-separated numbers, e.g., "100 101 102 ..."
    mapfile -t indices_array < <(seq "$chunk_actual_start" "$chunk_actual_end")
    
    if [[ ${#indices_array[@]} -eq 0 ]]; then
        log_message "Error: Failed to generate sequence of indices for range $chunk_actual_start to $chunk_actual_end."
        overall_script_success=1
        break # Stop processing if sequence generation fails
    fi

    # Define the output filename for this chunk
    output_filename="gemini-out-${chunk_actual_start}-${chunk_actual_end}.md"
    output_filepath="$OUTPUT_DIR/$output_filename"

    # Requirement 4: Echo processing message
    log_message "Processing file range $chunk_actual_start to $chunk_actual_end. Outputting to $output_filepath"

    # Execute the Python script with the generated indices
    # The output of the Python script (stdout) is redirected to the output file
    if ! "$PYTHON_EXECUTABLE" "$PYTHON_SCRIPT_PATH" "${indices_array[@]}" > "$output_filepath"; then
        # Python script exited with a non-zero status
        python_exit_code=$? # Capture the actual exit code
        log_message "Error: Python script failed for chunk $chunk_actual_start-$chunk_actual_end. Exit code: $python_exit_code."
        log_message "Output for this failed chunk (if any) is in $output_filepath."
        overall_script_success=1 # Mark overall script as failed
        break # Stop processing further chunks as per Requirement 2 logic
    else
        # Python script exited successfully (exit code 0) for this chunk
        log_message "Successfully processed chunk $chunk_actual_start-$chunk_actual_end."
        # Requirement 2: "it only returns success if all 100 designators are successfully processed."
        # This is interpreted as: the Python script for *this chunk* must exit with 0.
        # If the script needs to verify *each individual designator* within the Python output,
        # the Python script would need to provide a more granular success indicator,
        # or this bash script would need to parse the $output_filepath.
        # For now, a successful Python script exit for the chunk implies success for that chunk.
    fi
done

# --- Final Script Exit Status ---
if [[ $overall_script_success -eq 0 ]]; then
    log_message "All specified chunks processed successfully by the Python script."
    exit 0
else
    log_message "One or more chunks failed to process or an error occurred in the script."
    exit 1
fi

