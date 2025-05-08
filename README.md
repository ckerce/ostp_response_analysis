# Unified RFI Analyzer

This project provides a command-line tool to analyze Request for Information (RFI) responses. It extracts text from Markdown (`.md`) files stored within a tar archive, sends the content to a Google Gemini language model for analysis based on a structured prompt, and outputs the results in a Markdown table format. It also includes features for rate limiting, token management, and logging.

## Table of Contents

- [Project Overview](#project-overview)
- [Core Components](#core-components)
- [Input](#input)
- [Output](#output)
- [Processing Flow](#processing-flow)
- [Rate and Token Limit Enforcement](#rate-and-token-limit-enforcement)
  - [API Rate Limits (RPM, TPM)](#api-rate-limits-rpm-tpm)
  - [Document Processing Cycle Time](#document-processing-cycle-time)
  - [Input Text Truncation](#input-text-truncation)
- [Command-Line Arguments](#command-line-arguments)
- [Setup and Usage](#setup-and-usage)

## Project Overview

The primary goal of this tool is to automate the extraction of key information from a collection of RFI responses. It leverages Google's Gemini AI to understand and summarize documents, identify submitter details, key concerns, and policy recommendations.

## Core Components

The project consists of two main Python scripts:

1.  **`unified_rfi_analyzer.py`**:
    * The main executable script.
    * Handles command-line argument parsing.
    * Manages file operations, including reading from the tar archive.
    * Orchestrates the overall analysis workflow for each document.
    * Formats and prints the final output.
    * Controls the overall document processing pace.

2.  **`gemini_handler.py`**:
    * A utility class (`GeminiAPIHandler`) responsible for all interactions with the Google Gemini API.
    * Manages API key configuration.
    * Implements rate limiting for API calls (Requests Per Minute, Input/Output Tokens Per Minute) using a token bucket algorithm.
    * Counts tokens for input prompts.
    * Handles API call retries with exponential backoff for transient errors.
    * Parses JSON responses from the API.
    * Tracks API usage (requests, tokens, errors).

## Input

The script requires the following inputs:

1.  **Tar Archive**:
    * A `.tar` file containing the RFI responses.
    * The responses should be in Markdown (`.md`) format and located within a specified target directory inside the archive (default: `ostp_md/`).
    * Configured via the `--tarfile` argument (default: `ostp_responses.tar`).

2.  **Google API Key**:
    * A valid Google API key with access to the Generative Language API (Gemini).
    * Can be provided via the `--google_api_key` argument or read from an `api.key` file in the script's directory. The `api.key` file should contain the key on its first line.

3.  **File Indices (Positional Argument `K`)**:
    * One or more 0-based integer indices specifying which `.md` files from the tar archive (after listing and sorting) should be processed.
    * If omitted and `--list_files` is not used, the script will show an error and the help message.

## Output

The script produces several outputs:

1.  **Markdown Table (stdout)**:
    * The primary output, printed to the standard output.
    * Contains a table with columns: `Filename`, `Submitter Name`, `Submitter Type`, `Mission/Interest Summary`, `Key Concerns`, and `Policy Recommendations`.
    * Each processed document corresponds to a row in this table.

2.  **Application Log (`rfi_analyzer_app.log`)**:
    * A log file created in the script's directory.
    * Records information about the script's execution, including initialization, file processing steps, API call details, errors, and rate limiting actions.
    * Logging level can be configured (default: `INFO`).

3.  **Identified Sections Log (`identified_sections.log`)**:
    * A log file (default name, configurable via `--sections_log_file`).
    * Appends a JSON-formatted list of major sections identified by the LLM for each processed document.
    * Format: `doc_name : ["section1", "section2", ...]`

4.  **API Usage Summary (stderr & Log File)**:
    * At the end of processing, a summary of Gemini API usage is printed to standard error and also logged to `rfi_analyzer_app.log`.
    * Includes: model name, total requests made, total input tokens processed, total output tokens generated, and total API errors encountered.

## Processing Flow

1.  **Initialization**:
    * Parses command-line arguments.
    * Sets up logging.
    * Initializes the `GeminiAPIHandler` with the API key, model name, and rate limit parameters. This includes a test API call to verify connectivity and configuration.

2.  **File Discovery**:
    * Lists all `.md` files within the specified `target_dir` in the provided `tarfile`. Files are sorted for consistent indexing.
    * If `--list_files` is used, it prints the list of files with their indices and exits.

3.  **Document Iteration**:
    * Based on the provided `indices`, the script iterates through each selected file.
    * For each file:
        * **Extraction**: Text content is extracted from the tar archive.
        * **Cleaning**: Basic text cleaning is performed (e.g., consolidating multiple newlines).
        * **Truncation**: Text is truncated to `MAX_TEXT_LENGTH` (default: 80,000 characters) before being sent to the API.
        * **Analysis**:
            * A detailed prompt is constructed, instructing the Gemini model to extract specific information and respond in a JSON format.
            * The `GeminiAPIHandler.execute_analysis()` method is called. This method handles:
                * Counting input tokens.
                * Waiting for API rate limits (RPM, input/output TPM) if necessary.
                * Making the API call to `generate_content`.
                * Consuming tokens from rate limit buckets.
                * Parsing the JSON response.
                * Handling retries for API errors.
        * **Logging Sections**: If sections are identified in the analysis and a `sections_log_file` is specified, these are written to the log.
        * **Formatting Output**: The JSON analysis result is formatted into a Markdown table row.
        * **Printing Output**: The Markdown row is printed to `stdout`.
        * **Cycle Time Enforcement**: The script checks the time taken for the document. If it's less than `target_cycle_time_per_doc`, it sleeps for the remaining duration.

4.  **Completion**:
    * After processing all selected files, the script prints an "End of Analysis" message.
    * The `GeminiAPIHandler` provides an API usage summary, which is logged and printed to `stderr`.

## Rate and Token Limit Enforcement

The script employs several mechanisms to manage API usage and processing speed:

### API Rate Limits (RPM, TPM)

Managed by the `GeminiAPIHandler` using a `TokenBucket` class for each limit type:

* **Requests Per Minute (RPM)**: Limits the number of API calls made per minute.
* **Input Tokens Per Minute (Input TPM)**: Limits the total number of tokens sent to the API per minute.
* **Output Tokens Per Minute (Output TPM)**: Limits the total number of tokens received from the API per minute.

**How it works:**

1.  **Token Buckets**: Each limiter (RPM, Input TPM, Output TPM) is a token bucket. Buckets have a capacity (e.g., `rpm_limit`) and a fill rate (e.g., `rpm_limit / 60.0` tokens per second).
2.  **Token Counting**:
    * Before an API call, `_count_tokens()` determines the number of input tokens for the prompt.
    * An estimate for output tokens is also made for pre-flight checks.
3.  **Waiting Mechanism (`_wait_for_limits`)**:
    * Before making an API call, the handler checks if consuming 1 request, the calculated input tokens, and the estimated output tokens would exceed any bucket's current capacity.
    * If a limit would be hit, `get_wait_time()` calculates how long to wait for the buckets to refill sufficiently.
    * The script then `time.sleep()` for the maximum required wait time.
4.  **Token Consumption**:
    * If no wait is needed, tokens are "consumed" from the RPM and Input TPM buckets before the API call.
    * After a successful API call, the actual output tokens (obtained from `response.usage_metadata`) are consumed from the Output TPM bucket.
5.  **Retries**: If API calls fail due to rate limits (e.g., HTTP 429) or server-side issues (5xx), the handler implements an exponential backoff strategy before retrying, up to `max_retries`.

### Document Processing Cycle Time

* The `unified_rfi_analyzer.py` script has an argument `--target_cycle_time_per_doc` (default: 5 seconds).
* After each document is fully processed (including extraction, API call, and formatting), the script calculates the total time spent on that document.
* If this time is less than the `target_cycle_time_per_doc`, the script will `time.sleep()` for the difference.
* This ensures a minimum overall processing time per document, acting as an additional pacing mechanism independent of the direct API rate limits. It can be useful for smoothing out processing over longer periods or for systems that might have broader constraints.

### Input Text Truncation

* In `unified_rfi_analyzer.py`, the `MAX_TEXT_LENGTH` constant (default: 80,000 characters) defines a hard limit on the length of the text content sent to the LLM for analysis.
* If an extracted document's text exceeds this length, it is truncated before being included in the prompt. This is a pre-emptive measure to avoid overly long prompts that might exceed model context windows or consume excessive tokens unnecessarily.

## Command-Line Arguments

The script `unified_rfi_analyzer.py` accepts the following command-line arguments:

usage: unified_rfi_analyzer.py [-h] [--tarfile TARFILE] [--google_api_key GOOGLE_API_KEY] [--model MODEL] [--list_files] [--target_dir TARGET_DIR] [--rpm_limit RPM_LIMIT][--input_tpm_limit INPUT_TPM_LIMIT] [--output_tpm_limit OUTPUT_TPM_LIMIT] [--target_cycle_time_per_doc TARGET_CYCLE_TIME_PER_DOC][--sections_log_file SECTIONS_LOG_FILE] [--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}][K ...]Analyze RFI responses (.md files) from a tar archive using GeminiAPIHandler.positional arguments:K                             One or more 0-based indices of the files to process. If omitted and --list_files is not used, processes no files. (default: None)options:-h, --help                    show this help message and exit--tarfile TARFILE             Path to the tar archive. (default: ostp_responses.tar)--google_api_key GOOGLE_API_KEYGoogle API Key for Generative AI. If not provided here, it must be set as the GOOGLE_API_KEY environment variable or in 'api.key'.--model MODEL                 Google AI model name (e.g., gemini-2.0-flash). (default: gemini-2.0-flash)--list_files                  List all .md files found in the archive's target directory and exit. (default: False)--target_dir TARGET_DIR       Target directory within the tar archive. (default: ostp_md/)--rpm_limit RPM_LIMIT         Requests Per Minute limit for the API. (default: 15)--input_tpm_limit INPUT_TPM_LIMITInput Tokens Per Minute limit for the API. (default: 1000000)--output_tpm_limit OUTPUT_TPM_LIMITOutput Tokens Per Minute limit for the API. (default: 32000)--target_cycle_time_per_doc TARGET_CYCLE_TIME_PER_DOCOptional: Minimum overall cycle time in seconds per document. API rate limiting is primarily handled by RPM/TPM. (default: 5.0)--sections_log_file SECTIONS_LOG_FILEFile to append identified document sections to. Format: doc_name : [sections_list] (default: identified_sections.log)--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}Set the logging level for the application and handler. (default: INFO)
## Setup and Usage

1.  **Prerequisites**:
    * Python 3.x
    * Required Python packages: `google-generativeai` (install via `pip install google-generativeai`)

2.  **API Key**:
    * Ensure you have a Google API Key for the Generative AI services.
    * Place it in a file named `api.key` in the same directory as the scripts (one key per line, ensure no extra spaces/newlines), or provide it via the `--google_api_key` argument, or set it as the `GOOGLE_API_KEY` environment variable.

3.  **Prepare Input**:
    * Have your RFI responses as `.md` files.
    * Package them into a tar archive (e.g., `ostp_responses.tar`). Ensure the `.md` files are within a subdirectory inside the tar file that matches the `--target_dir` argument (default `ostp_md/`).

4.  **Running the script**:

    * **List files**:
        ```bash
        python unified_rfi_analyzer.py --list_files --tarfile path/to/your/archive.tar
        ```
        Note the indices of the files you want to process.

    * **Analyze specific files**:
        ```bash
        python unified_rfi_analyzer.py --tarfile path/to/your/archive.tar <index1> <index2> ...
        ```
        Example:
        ```bash
        python unified_rfi_analyzer.py 0 1 5
        ```

    * **Redirect output**:
        To save the Markdown table to a file:
        ```bash
        python unified_rfi_analyzer.py <indices> > analysis_results.md
        ```

    * **Adjusting rate limits and logging**:
        Use the respective command-line arguments (e.g., `--rpm_limit 10`, `--loglevel DEBUG`).

