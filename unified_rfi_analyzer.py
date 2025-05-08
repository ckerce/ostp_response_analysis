# -- ./unified_rfi_analyzer.py
import tarfile
import io
import re
import os
import argparse
import sys
import json
import time
import logging # For configuring logging

# Import from your new utility file
from gemini_handler import GeminiAPIHandler # Assuming gemini_handler.py is in the same directory or Python path

# --- Configuration ---
DEFAULT_TAR_FILE_PATH = 'ostp_responses.tar'
with open('api.key','r') as f:
    DEFAULT_GOOGLE_API_KEY=f.readline().strip()
#DEFAULT_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEFAULT_MODEL_NAME = 'gemini-2.0-flash' # User updated
TARGET_DIR_IN_TAR = 'ostp_md/'
MAX_TEXT_LENGTH = 80000 # Character limit for text sent to LLM

# This is less critical if handler manages rates precisely,
# but can be kept as a minimum overall processing time per doc if desired for other pacing.
DEFAULT_TARGET_DOCUMENT_CYCLE_SECONDS = 5 # User updated
DEFAULT_SECTIONS_LOG_FILE = "identified_sections.log" # From previous canvas version

# --- Logging Setup ---
# Configure logging once at the application level
# This basicConfig will apply to all loggers unless they are specifically configured otherwise.
logging.basicConfig(
    level=logging.INFO, # Default level for all loggers, can be overridden by CLI arg or for specific modules
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rfi_analyzer_app.log", mode='a'), # Application log for this script's execution
        logging.StreamHandler(sys.stderr) # Also print INFO and above to stderr
    ]
)
# Example: Set a different level for the gemini_handler if too verbose/quiet
# logging.getLogger('gemini_handler').setLevel(logging.DEBUG)


# --- Helper Functions (largely unchanged) ---

def list_md_files_in_tar(tar_path, target_dir):
    """
    Lists all .md files within the target directory in the tar archive.
    Sorts the list for consistent indexing.
    """
    logger = logging.getLogger(__name__)
    md_files = []
    try:
        with tarfile.open(tar_path, 'r') as tar:
            all_members = tar.getmembers()
            logger.info(f"Scanning {len(all_members)} members in '{tar_path}'...")
            for member in all_members:
                normalized_member_name = os.path.normpath(member.name)
                normalized_target_dir = os.path.normpath(target_dir)
                if normalized_member_name.startswith(normalized_target_dir) and \
                   normalized_member_name.lower().endswith('.md') and \
                   member.isfile():
                    md_files.append(member.name)
        md_files.sort()
        logger.info(f"Found {len(md_files)} '.md' files in '{target_dir}'.")
        return md_files
    except FileNotFoundError:
        logger.error(f"Tar file not found at '{tar_path}'. Please check the path.")
        return None
    except tarfile.ReadError as e:
        logger.error(f"Could not read tar file '{tar_path}'. It might be corrupted or not a tar file: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading tar file: {e}")
        return None

def extract_text_from_tar(tar_path, file_path_in_tar):
    """
    Extracts text content of a specific file from the tar archive.
    """
    logger = logging.getLogger(__name__)
    try:
        with tarfile.open(tar_path, 'r') as tar:
            member = tar.getmember(file_path_in_tar)
            file_obj = tar.extractfile(member)
            if file_obj:
                content = file_obj.read().decode('utf-8', errors='ignore')
                return content
            else:
                logger.error(f"Could not get file object for '{file_path_in_tar}' within the tar.")
                return None
    except KeyError:
        logger.error(f"File not found in tar archive: '{file_path_in_tar}'")
        return None
    except FileNotFoundError: # Should be caught by list_md_files_in_tar already
        logger.error(f"Tar file not found at '{tar_path}'.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred extracting file '{file_path_in_tar}': {e}")
        return None

def clean_text(text):
    """
    Basic text cleaning: removes excessive newlines.
    """
    if not text:
        return ""
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', text) # Consolidate multiple newlines
    return cleaned_text.strip()

def analyze_document_with_handler(api_handler, text_content, filename):
    """
    Prepares the prompt and uses the GeminiAPIHandler to get analysis.
    Args:
        api_handler (GeminiAPIHandler): Instance of the API handler.
        text_content (str): The cleaned text content of the RFI response.
        filename (str): The original filename (used in the prompt).
    Returns:
        dict: A dictionary containing the parsed analysis from the handler.
    """
    logger = logging.getLogger(__name__)
    truncated_indicator = "..." if len(text_content) > MAX_TEXT_LENGTH else ""
    content_to_send = text_content[:MAX_TEXT_LENGTH]

    # This is the unified prompt template
    unified_prompt = f"""
Analyze the RFI response text from filename '{filename}'.
The response is to an OSTP RFI on removing barriers to American leadership in AI.
Extract information based *only* on the provided text.

**Output Format:**
Respond with a single, valid JSON object. Do NOT include any text outside this JSON object (e.g. no "```json" or "```" markers, just the raw JSON).
The JSON object should have the following top-level keys:
- "submitter_name": string (Extract submitter's name. Examples: Georgia Tech, Verizon, Anonymous, Specific Person Name. Default to "Not Stated" if not found.)
- "inferred_submitter_type": string (Categorize. Examples: Academia, Industry-Large, Industry-Startup, Industry-Association, Professional-Society, Advocacy/Think Tank, Government, Individual, Anonymous, Unclear. Default to "Unclear" if not determinable.)
- "mission_interest_summary": string (1-2 sentence summary of submitter's AI policy mission/focus from the document. Default to "Not Stated" if not found.)
- "key_concerns": array of strings (List main concerns, barriers, or risks. If the document has clear sections that you identify, you MAY optionally prefix each concern with "[From Section X: Section Name]" if relevant and helpful for context. Use a new string for each distinct concern. If no concerns are explicitly mentioned, use an empty array [] or an array with a single string "None Stated".)
- "policy_recommendations": array of strings (List main policy recommendations. Similar to concerns, you MAY optionally prefix with section information if relevant. Use a new string for each distinct recommendation. If no recommendations are explicitly proposed, use an empty array [] or an array with a single string "None Stated".)
- "identified_sections": array of strings (Optional: List names or brief descriptions of major sections you identified, e.g., ["Introduction", "Challenges Regarding Data Access", "Recommendations for Workforce Development"]. If no clear sections are evident or you don't base your primary analysis on them, this can be an empty array [].)

**RFI Response Text (first {MAX_TEXT_LENGTH} characters if truncated):**
```markdown
{content_to_send}{truncated_indicator}
```

**Important Notes:**
- Ensure all string values within the JSON are properly escaped if they contain special characters (like quotes or newlines).
- For "key_concerns" and "policy_recommendations", if none are found, prefer an empty array `[]`.
- Do not invent information. Stick strictly to the document content.

Begin the JSON object now:
"""
    logger.info(f"Preparing to analyze document: {filename} with GeminiAPIHandler.")
    
    # The API call is delegated to the handler
    analysis_result_dict = api_handler.execute_analysis(unified_prompt)

    # Basic validation of returned structure from handler (it should return a default error dict if failed)
    core_keys = ["submitter_name", "inferred_submitter_type", "mission_interest_summary", "key_concerns", "policy_recommendations"]
    if not all(k in analysis_result_dict for k in core_keys):
        logger.warning(
            f"Analysis for {filename} from handler returned data with missing core keys. "
            f"Data: {str(analysis_result_dict)[:200]}"
        )
        # Ensure expected list structures exist for downstream processing, even if it's an error dict
        analysis_result_dict.setdefault("key_concerns", [])
        analysis_result_dict.setdefault("policy_recommendations", [])
        analysis_result_dict.setdefault("identified_sections", [])
        analysis_result_dict.setdefault("submitter_name", "Parsing/Structure Error Post-Handler") # Ensure key exists

    return analysis_result_dict

def format_as_markdown_row(filename, analysis_data):
    """
    Formats the analysis result dictionary into a Markdown table row.
    (Largely unchanged, but expects analysis_data to be a dictionary)
    """
    logger = logging.getLogger(__name__)
    if not isinstance(analysis_data, dict): 
        logger.error(f"Invalid analysis_data type for {filename}: {type(analysis_data)}. Using error defaults.")
        analysis_data = {
            "submitter_name": "Formatting Error", "inferred_submitter_type": "Invalid data type",
            "mission_interest_summary": "N/A", "key_concerns": ["N/A"],
            "policy_recommendations": ["N/A"], "identified_sections": []
        }

    s_name = str(analysis_data.get("submitter_name", "Not Stated"))
    s_type = str(analysis_data.get("inferred_submitter_type", "Unclear"))
    s_summary = str(analysis_data.get("mission_interest_summary", "Not Stated"))

    concerns_list = analysis_data.get("key_concerns", [])
    if not concerns_list or (len(concerns_list) == 1 and str(concerns_list[0]).lower() == "none stated"):
        formatted_concerns = "- None Stated"
    else:
        valid_concerns = [str(item).strip() for item in concerns_list if str(item).strip() and str(item).strip().lower() != "none stated"]
        if not valid_concerns: formatted_concerns = "- None Stated"
        else: formatted_concerns = "<br>".join(f"- {item}" for item in valid_concerns)

    recommendations_list = analysis_data.get("policy_recommendations", [])
    if not recommendations_list or (len(recommendations_list) == 1 and str(recommendations_list[0]).lower() == "none stated"):
        formatted_recommendations = "- None Stated"
    else:
        valid_recommendations = [str(item).strip() for item in recommendations_list if str(item).strip() and str(item).strip().lower() != "none stated"]
        if not valid_recommendations: formatted_recommendations = "- None Stated"
        else: formatted_recommendations = "<br>".join(f"- {item}" for item in valid_recommendations)

    base_filename = os.path.basename(filename).replace('|', '\\|') # Escape pipes for Markdown
    s_name = s_name.replace('|', '\\|')
    s_type = s_type.replace('|', '\\|')
    s_summary = s_summary.replace('|', '\\|')
    # Concerns/Recommendations use <br>; internal pipes should ideally be handled by LLM or are rare.

    return f"| {base_filename} | {s_name} | {s_type} | {s_summary} | {formatted_concerns} | {formatted_recommendations} |"

# --- Main Execution ---
if __name__ == "__main__":
    main_logger = logging.getLogger(__name__) # Logger for the main execution block

    parser = argparse.ArgumentParser(
        description="Analyze RFI responses (.md files) from a tar archive using GeminiAPIHandler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("indices", metavar="K", type=int, nargs='*',
                        help="One or more 0-based indices of the files to process. If omitted and --list_files is not used, processes no files.")
    parser.add_argument("--tarfile", default=DEFAULT_TAR_FILE_PATH, help="Path to the tar archive.")
    parser.add_argument("--google_api_key", default=DEFAULT_GOOGLE_API_KEY,
                        help="Google API Key for Generative AI. If not provided here, it must be set as the GOOGLE_API_KEY environment variable.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Google AI model name (e.g., gemini-2.0-flash).")
    parser.add_argument("--list_files", action="store_true", help="List all .md files found in the archive's target directory and exit.")
    parser.add_argument("--target_dir", default=TARGET_DIR_IN_TAR, help="Target directory within the tar archive.")
    
    # Rate limit arguments for the handler
    parser.add_argument("--rpm_limit", type=int, default=15, help="Requests Per Minute limit for the API.")
    parser.add_argument("--input_tpm_limit", type=int, default=1000000, help="Input Tokens Per Minute limit for the API.")
    parser.add_argument("--output_tpm_limit", type=int, default=32000, help="Output Tokens Per Minute limit for the API.")
    
    parser.add_argument("--target_cycle_time_per_doc", type=float, default=DEFAULT_TARGET_DOCUMENT_CYCLE_SECONDS,
                        help=f"Optional: Minimum overall cycle time in seconds per document. API rate limiting is primarily handled by RPM/TPM. Default: {DEFAULT_TARGET_DOCUMENT_CYCLE_SECONDS}s.")
    parser.add_argument("--sections_log_file", default=DEFAULT_SECTIONS_LOG_FILE,
                        help="File to append identified document sections to. Format: doc_name : [sections_list]")
    parser.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level for the application and handler.")


    args = parser.parse_args()

    # Update logging level based on CLI argument
    # This affects all loggers unless they have their own level set explicitly.
    try:
        # Get the root logger and set its level.
        # Also, set the level for the stream handler to ensure console output reflects the choice.
        logging.getLogger().setLevel(args.loglevel.upper())
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.StreamHandler): # Target the console handler
                handler.setLevel(args.loglevel.upper())
        main_logger.info(f"Logging level set to {args.loglevel.upper()}")
    except ValueError:
        main_logger.error(f"Invalid log level: {args.loglevel}. Using INFO.")
        logging.getLogger().setLevel(logging.INFO)


    if not args.google_api_key:
        main_logger.critical("Google API Key not found. Please provide it via --google_api_key or set GOOGLE_API_KEY environment variable.")
        sys.exit(1)

    # Initialize the API Handler
    try:
        api_handler = GeminiAPIHandler(
            api_key=args.google_api_key,
            model_name=args.model,
            rpm_limit=args.rpm_limit,
            input_tpm_limit=args.input_tpm_limit,
            output_tpm_limit=args.output_tpm_limit
            # max_retries can also be an arg if desired
        )
        main_logger.info("Performing a quick test call via GeminiAPIHandler...")
        test_prompt = 'Test: Briefly say hello. Respond ONLY with a valid JSON object containing a single key "greeting" and your greeting as the value. For example: {"greeting": "Hello!"}'
        # The handler's execute_analysis expects only the prompt
        test_response_dict = api_handler.execute_analysis(test_prompt) 
        
        # Check if the response is the default error structure from the handler
        if test_response_dict.get("submitter_name") == "API Handler Error":
             raise Exception(f"Test call via API Handler failed. Handler response: {test_response_dict.get('mission_interest_summary', 'Unknown error from handler')}")
        main_logger.info(f"Successfully initialized and tested GeminiAPIHandler. Test response (dict format): {str(test_response_dict)[:100]}...")

    except Exception as e:
        main_logger.critical(f"Failed to initialize or test GeminiAPIHandler: {e}", exc_info=True)
        sys.exit(1)

    all_files = list_md_files_in_tar(args.tarfile, args.target_dir)
    if all_files is None: sys.exit(1) # Error already logged by function
    if not all_files:
        main_logger.warning(f"No '.md' files found in '{args.target_dir}' within '{args.tarfile}'.")
        if not args.list_files: sys.exit(0)

    if args.list_files:
        if all_files:
            print("\nFiles found (Index: Path):")
            for i, f_path in enumerate(all_files): print(f"{i}: {f_path}")
        else: print(f"\nNo '.md' files found in '{args.target_dir}' within '{args.tarfile}'.")
        sys.stdout.flush()
        sys.exit(0)

    if not args.indices:
        main_logger.error("No file indices provided for analysis. Use --list_files to see available indices.")
        parser.print_help(sys.stderr); sys.stderr.flush(); sys.exit(1)
    
    files_to_process = []
    invalid_indices = []
    if all_files: 
        max_index = len(all_files) - 1
        for k_idx in args.indices:
            if 0 <= k_idx <= max_index: files_to_process.append(all_files[k_idx])
            else: invalid_indices.append(k_idx)
    else: invalid_indices.extend(args.indices) # If no files, all indices are invalid

    if invalid_indices:
        main_logger.error(f"Invalid index/indices provided: {invalid_indices}")
        if all_files: main_logger.error(f"Valid indices range from 0 to {len(all_files) - 1}.")
        else: main_logger.error("No files available to index.")
        sys.exit(1)
    if not files_to_process:
        main_logger.error("No valid file indices resulted in files to process.")
        sys.exit(1)

    # Print Markdown table header
    print("| Filename | Submitter Name | Submitter Type | Mission/Interest Summary | Key Concerns | Policy Recommendations |")
    print("|---|---|---|---|---|---|")
    sys.stdout.flush() # Ensure header is printed before any potential delays

    for file_path_in_tar in files_to_process:
        document_processing_start_time = time.monotonic()
        main_logger.info(f"Processing: {file_path_in_tar}...")
        
        text_content = extract_text_from_tar(args.tarfile, file_path_in_tar)
        analysis_data_dict = None # Initialize

        if text_content is None:
            main_logger.warning(f"Skipping {file_path_in_tar} due to extraction error.")
            analysis_data_dict = {
                "submitter_name": "Extraction Error", "inferred_submitter_type": "N/A",
                "mission_interest_summary": "N/A", "key_concerns": [], 
                "policy_recommendations": [], "identified_sections": []
            }
        else:
            cleaned_text = clean_text(text_content)
            if not cleaned_text:
                main_logger.warning(f"Skipping {file_path_in_tar} as content is empty after cleaning.")
                analysis_data_dict = {
                    "submitter_name": "Empty Content", "inferred_submitter_type": "N/A",
                    "mission_interest_summary": "N/A", "key_concerns": [], 
                    "policy_recommendations": [], "identified_sections": []
                }
            else:
                main_logger.info(f"Extracted and cleaned text (length: {len(cleaned_text)} chars) for {file_path_in_tar}.")
                analysis_data_dict = analyze_document_with_handler(
                    api_handler,
                    cleaned_text,
                    file_path_in_tar
                )
                
                # Log identified sections if present and a log file is specified
                if args.sections_log_file and analysis_data_dict and "identified_sections" in analysis_data_dict:
                    sections = analysis_data_dict.get("identified_sections")
                    # Only log if sections is a non-empty list and not the error default from handler
                    if sections and not (analysis_data_dict.get("submitter_name") == "API Handler Error"): 
                        base_doc_name = os.path.basename(file_path_in_tar)
                        try:
                            with open(args.sections_log_file, 'a', encoding='utf-8') as slog:
                                slog.write(f"{base_doc_name} : {json.dumps(sections)}\n") # Use json.dumps for proper list formatting
                            main_logger.info(f"Logged identified sections to '{args.sections_log_file}'. Sections: {sections}")
                        except IOError as e:
                            main_logger.warning(f"Could not write to sections log file '{args.sections_log_file}': {e}")
                    elif not (analysis_data_dict.get("submitter_name") == "API Handler Error" or \
                              analysis_data_dict.get("submitter_name") == "Extraction Error" or \
                              analysis_data_dict.get("submitter_name") == "Empty Content"):
                        main_logger.info(f"No sections identified by LLM for {file_path_in_tar} to log.")

        markdown_row = format_as_markdown_row(file_path_in_tar, analysis_data_dict)
        print(markdown_row) # Print to stdout for table
        sys.stdout.flush()

        # Optional: Keep target_cycle_time_per_doc if you want an *additional* overall delay
        time_spent_this_document = time.monotonic() - document_processing_start_time
        wait_needed_for_cycle_target = args.target_cycle_time_per_doc - time_spent_this_document
        
        if wait_needed_for_cycle_target > 0:
            main_logger.info(f"Document processing and API calls took {time_spent_this_document:.2f}s. Min cycle time enforced: Waiting for {wait_needed_for_cycle_target:.2f}s.")
            time.sleep(wait_needed_for_cycle_target)
        else:
            main_logger.info(f"Document processing and API calls took {time_spent_this_document:.2f}s. (Min cycle target: {args.target_cycle_time_per_doc}s).")

    main_logger.info("--- End of Analysis ---")
    usage_summary = api_handler.get_usage_summary()
    main_logger.info(f"Gemini API Usage Summary by Handler: {usage_summary}")
    # Also print summary to stderr for easy viewing
    print(f"\n[Info] Gemini API Usage Summary by Handler: {json.dumps(usage_summary, indent=2)}", file=sys.stderr)
