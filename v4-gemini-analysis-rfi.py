#!/usr/bin/env python3
import tarfile
import io
import re
import os
import argparse
import sys
import json
import time # Added for rate limiting
import google.generativeai as genai

# --- Configuration ---
DEFAULT_TAR_FILE_PATH = 'ostp_responses.tar'
DEFAULT_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#DEFAULT_MODEL_NAME = 'gemini-2.0-flash' # User updated
DEFAULT_MODEL_NAME = 'gemini-1.5-flash' # User updated
TARGET_DIR_IN_TAR = 'ostp_md/'
MAX_TEXT_LENGTH = 80000 # Character limit for text sent to LLM

# Default target cycle time per document in seconds.
DEFAULT_TARGET_DOCUMENT_CYCLE_SECONDS = 5 # User updated
DEFAULT_SECTIONS_LOG_FILE = "ostp-gemini-identified-sections.log" # From canvas version

# --- Helper Functions ---

def list_md_files_in_tar(tar_path, target_dir):
    """
    Lists all .md files within the target directory in the tar archive.
    Sorts the list for consistent indexing.
    Args:
        tar_path (str): Path to the .tar archive.
        target_dir (str): The directory prefix within the tar file (e.g., 'ostp_md/').
    Returns:
        list: A sorted list of full paths to .md files within the tar archive, or None on error.
    """
    md_files = []
    try:
        with tarfile.open(tar_path, 'r') as tar:
            all_members = tar.getmembers()
            print(f"[Info] Scanning {len(all_members)} members in '{tar_path}'...", file=sys.stderr)
            sys.stderr.flush()
            for member in all_members:
                normalized_member_name = os.path.normpath(member.name)
                normalized_target_dir = os.path.normpath(target_dir)
                if normalized_member_name.startswith(normalized_target_dir) and \
                   normalized_member_name.lower().endswith('.md') and \
                   member.isfile():
                    md_files.append(member.name)
        md_files.sort()
        print(f"[Info] Found {len(md_files)} '.md' files in '{target_dir}'.", file=sys.stderr)
        sys.stderr.flush()
        return md_files
    except FileNotFoundError:
        print(f"[Error] Tar file not found at '{tar_path}'. Please check the path.", file=sys.stderr)
        sys.stderr.flush()
        return None
    except tarfile.ReadError as e:
        print(f"[Error] Could not read tar file '{tar_path}'. It might be corrupted or not a tar file: {e}", file=sys.stderr)
        sys.stderr.flush()
        return None
    except Exception as e:
        print(f"[Error] An unexpected error occurred while reading tar file: {e}", file=sys.stderr)
        sys.stderr.flush()
        return None

def extract_text_from_tar(tar_path, file_path_in_tar):
    """
    Extracts text content of a specific file from the tar archive.
    Args:
        tar_path (str): Path to the .tar archive.
        file_path_in_tar (str): The full path of the file inside the tar archive.
    Returns:
        str: The decoded text content of the file, or None on error.
    """
    try:
        with tarfile.open(tar_path, 'r') as tar:
            member = tar.getmember(file_path_in_tar)
            file_obj = tar.extractfile(member)
            if file_obj:
                content = file_obj.read().decode('utf-8', errors='ignore')
                return content
            else:
                print(f"[Error] Could not get file object for '{file_path_in_tar}' within the tar.", file=sys.stderr)
                sys.stderr.flush()
                return None
    except KeyError:
        print(f"[Error] File not found in tar archive: '{file_path_in_tar}'", file=sys.stderr)
        sys.stderr.flush()
        return None
    except FileNotFoundError:
        print(f"[Error] Tar file not found at '{tar_path}'.", file=sys.stderr)
        sys.stderr.flush()
        return None
    except Exception as e:
        print(f"[Error] An unexpected error occurred extracting file '{file_path_in_tar}': {e}", file=sys.stderr)
        sys.stderr.flush()
        return None

def clean_text(text):
    """
    Basic text cleaning: removes excessive newlines.
    Args:
        text (str): The raw text content.
    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', text)
    return cleaned_text.strip()

def analyze_document_unified_google_ai(gemini_model, text_content, filename):
    """
    Analyzes the document structure and content using a single Google AI API call.
    Args:
        gemini_model: An initialized google.generativeai.GenerativeModel instance.
        text_content (str): The cleaned text content of the RFI response.
        filename (str): The original filename (used in the prompt).
    Returns:
        dict: A dictionary containing the parsed analysis, or a default error structure.
    """
    truncated_indicator = "..." if len(text_content) > MAX_TEXT_LENGTH else ""
    content_to_send = text_content[:MAX_TEXT_LENGTH]

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
    default_error_output = {
        "submitter_name": "Analysis Error",
        "inferred_submitter_type": "Analysis Error",
        "mission_interest_summary": "Error during unified analysis.",
        "key_concerns": [],
        "policy_recommendations": [],
        "identified_sections": []
    }
    response_text = None # Initialize for debugging

    try:
        print(f"  [Info] Sending unified analysis request to Google AI ({gemini_model.model_name})...", file=sys.stderr)
        sys.stderr.flush()
        
        response = gemini_model.generate_content(unified_prompt)
        
        print(f"  [Info] Received unified analysis response from Google AI.", file=sys.stderr)
        sys.stderr.flush()

        response_text = response.text
        
        json_text = response_text.strip()
        if json_text.startswith("```json"): 
            json_text = re.sub(r"^```json\s*|\s*```$", "", json_text, flags=re.DOTALL)
        json_text = json_text.strip()

        parsed_data = json.loads(json_text)
        
        core_keys = ["submitter_name", "inferred_submitter_type", "mission_interest_summary", "key_concerns", "policy_recommendations"]
        if not all(k in parsed_data for k in core_keys):
            print(f"  [Warning] Unified analysis returned JSON with missing core keys for {filename}. Defaulting.", file=sys.stderr)
            print(f"  [Debug] Received JSON structure (first 200 chars): {json_text[:200]}...", file=sys.stderr)
            sys.stderr.flush()
            return default_error_output
        
        if not isinstance(parsed_data.get("key_concerns"), list):
            parsed_data["key_concerns"] = [str(parsed_data.get("key_concerns"))] if parsed_data.get("key_concerns") else []
        if not isinstance(parsed_data.get("policy_recommendations"), list):
            parsed_data["policy_recommendations"] = [str(parsed_data.get("policy_recommendations"))] if parsed_data.get("policy_recommendations") else []
            
        return parsed_data

    except json.JSONDecodeError as e:
        print(f"  [Warning] Could not parse unified analysis response as valid JSON for {filename}: {e}", file=sys.stderr)
        if response_text:
            print(f"  [Debug] Received text for JSON parsing (first 500 chars): {response_text[:500]}...", file=sys.stderr)
        sys.stderr.flush()
        return default_error_output
    except Exception as e:
        print(f"  [Error] Error during unified analysis with Google AI for {filename}: {e}", file=sys.stderr)
        if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            print(f"  [Error] Prompt Feedback: {response.prompt_feedback}", file=sys.stderr)
        elif response_text: 
             print(f"  [Debug] Raw response text (first 200 chars): {response_text[:200]}...", file=sys.stderr)
        sys.stderr.flush()
        return default_error_output

def format_as_markdown_row(filename, analysis_data):
    """
    Formats the analysis result dictionary into a Markdown table row.
    Args:
        filename (str): The original filename of the response.
        analysis_data (dict): The dictionary output from analyze_document_unified_google_ai.
    Returns:
        str: A formatted Markdown table row string.
    """
    if not isinstance(analysis_data, dict): 
        analysis_data = {
            "submitter_name": "Formatting Error", "inferred_submitter_type": "Invalid data",
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
        if not valid_concerns:
            formatted_concerns = "- None Stated"
        else:
            formatted_concerns = "<br>".join(f"- {item}" for item in valid_concerns)

    recommendations_list = analysis_data.get("policy_recommendations", [])
    if not recommendations_list or (len(recommendations_list) == 1 and str(recommendations_list[0]).lower() == "none stated"):
        formatted_recommendations = "- None Stated"
    else:
        valid_recommendations = [str(item).strip() for item in recommendations_list if str(item).strip() and str(item).strip().lower() != "none stated"]
        if not valid_recommendations:
            formatted_recommendations = "- None Stated"
        else:
            formatted_recommendations = "<br>".join(f"- {item}" for item in valid_recommendations)

    base_filename = os.path.basename(filename).replace('|', '\\|')
    s_name = s_name.replace('|', '\\|')
    s_type = s_type.replace('|', '\\|')
    s_summary = s_summary.replace('|', '\\|')

    return f"| {base_filename} | {s_name} | {s_type} | {s_summary} | {formatted_concerns} | {formatted_recommendations} |"

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze RFI responses (.md files) from a tar archive using a single Google Generative AI call per document.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("indices", metavar="K", type=int, nargs='*',
                        help="One or more 0-based indices of the files to process. If omitted and --list_files is not used, processes no files.")
    parser.add_argument("--tarfile", default=DEFAULT_TAR_FILE_PATH,
                        help="Path to the tar archive.")
    parser.add_argument("--google_api_key", default=DEFAULT_GOOGLE_API_KEY,
                        help="Google API Key for Generative AI. If not provided here, it must be set as the GOOGLE_API_KEY environment variable.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help="Google AI model name (e.g., gemini-1.5-flash-latest, gemini-pro).")
    parser.add_argument("--list_files", action="store_true",
                        help="List all .md files found in the archive's target directory and exit.")
    parser.add_argument("--target_dir", default=TARGET_DIR_IN_TAR,
                        help="Target directory within the tar archive.")
    parser.add_argument("--target_cycle_time_per_doc", type=float, default=DEFAULT_TARGET_DOCUMENT_CYCLE_SECONDS,
                        help=f"Target cycle time in seconds per document (including API call and any added sleep) to manage API rate limits. Default: {DEFAULT_TARGET_DOCUMENT_CYCLE_SECONDS}s for one main API call.")
    parser.add_argument("--sections_log_file", default=DEFAULT_SECTIONS_LOG_FILE, # From canvas version
                        help="File to append identified document sections to. Format: doc_name : [sections_list]")


    args = parser.parse_args()

    if not args.google_api_key:
        print("[Error] Google API Key not found. Please provide it via --google_api_key or set GOOGLE_API_KEY environment variable.", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
    try:
        genai.configure(api_key=args.google_api_key)
        print("[Info] Google AI SDK configured with API key.", file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:
        print(f"[Error] Failed to configure Google AI SDK: {e}", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

    all_files = list_md_files_in_tar(args.tarfile, args.target_dir)
    if all_files is None:
        sys.exit(1) 
    if not all_files:
        print(f"[Warning] No '.md' files found in '{args.target_dir}' within '{args.tarfile}'.", file=sys.stderr)
        sys.stderr.flush()
        if not args.list_files:
            sys.exit(0)

    if args.list_files:
        if all_files:
            print("\nFiles found (Index: Path):")
            for i, f_path in enumerate(all_files):
                print(f"{i}: {f_path}")
        else:
            print(f"\nNo '.md' files found in '{args.target_dir}' within '{args.tarfile}'.")
        sys.stdout.flush()
        sys.exit(0)

    if not args.indices:
        print("[Error] No file indices provided for analysis. Use --list_files to see available indices.", file=sys.stderr)
        sys.stderr.flush()
        parser.print_help(sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

    files_to_process = []
    invalid_indices = []
    if all_files: 
        max_index = len(all_files) - 1
        for k_idx in args.indices:
            if 0 <= k_idx <= max_index:
                files_to_process.append(all_files[k_idx])
            else:
                invalid_indices.append(k_idx)
    else: 
        invalid_indices.extend(args.indices)


    if invalid_indices:
        print(f"\n[Error] Invalid index/indices provided: {invalid_indices}", file=sys.stderr)
        if all_files:
            print(f"Valid indices range from 0 to {len(all_files) - 1}.", file=sys.stderr)
        else:
            print("No files available to index.", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

    if not files_to_process:
        print("[Error] No valid file indices resulted in files to process.", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

    try:
        print(f"\n[Info] Attempting to initialize Google AI model: {args.model}...", file=sys.stderr)
        sys.stderr.flush()
        model = genai.GenerativeModel(args.model)
        print("[Info] Performing a quick test call to the model...", file=sys.stderr)
        sys.stderr.flush()
        test_response = model.generate_content("Test: Briefly say hello.", generation_config=genai.types.GenerationConfig(max_output_tokens=20))
        
        if not test_response.text:
            feedback_info = ""
            if hasattr(test_response, 'prompt_feedback') and test_response.prompt_feedback:
                feedback_info = f" Feedback: {test_response.prompt_feedback}"
            elif hasattr(test_response, 'candidates') and not test_response.candidates:
                feedback_info = " Feedback: No candidates returned."
            raise Exception(f"Test call to model '{args.model}' failed to return text.{feedback_info}")
        print(f"[Info] Successfully initialized and tested Google AI model '{args.model}'. Test response: {test_response.text[:50].strip()}...", file=sys.stderr)
        sys.stderr.flush()
    except Exception as e:
        print(f"\n[Error] Failed to initialize or test Google AI model '{args.model}': {e}", file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

    print("| Filename | Submitter Name | Submitter Type | Mission/Interest Summary | Key Concerns | Policy Recommendations |")
    print("|---|---|---|---|---|---|")
    sys.stdout.flush()

    target_document_cycle_seconds = args.target_cycle_time_per_doc

    for file_path_in_tar in files_to_process:
        document_processing_start_time = time.monotonic()

        print(f"\nProcessing: {file_path_in_tar}...", file=sys.stderr)
        sys.stderr.flush()
        text_content = extract_text_from_tar(args.tarfile, file_path_in_tar)
        
        analysis_data_dict = None 

        if text_content is None:
            print(f"  Skipping {file_path_in_tar} due to extraction error.", file=sys.stderr)
            sys.stderr.flush()
            analysis_data_dict = {
                "submitter_name": "Extraction Error", "inferred_submitter_type": "N/A",
                "mission_interest_summary": "N/A", "key_concerns": [], "policy_recommendations": [],
                "identified_sections": [] 
            }
        else:
            cleaned_text = clean_text(text_content)
            if not cleaned_text:
                print(f"  Skipping {file_path_in_tar} as content is empty after cleaning.", file=sys.stderr)
                sys.stderr.flush()
                analysis_data_dict = {
                    "submitter_name": "Empty Content", "inferred_submitter_type": "N/A",
                    "mission_interest_summary": "N/A", "key_concerns": [], "policy_recommendations": [],
                    "identified_sections": [] 
                }
            else:
                print(f"  Extracted and cleaned text (length: {len(cleaned_text)} chars).", file=sys.stderr)
                sys.stderr.flush()
                
                analysis_data_dict = analyze_document_unified_google_ai(
                    model,
                    cleaned_text,
                    file_path_in_tar
                )
                # Log identified sections if present and a log file is specified (From canvas version)
                if args.sections_log_file and analysis_data_dict and "identified_sections" in analysis_data_dict:
                    sections = analysis_data_dict.get("identified_sections")
                    if sections: 
                        base_doc_name = os.path.basename(file_path_in_tar)
                        try:
                            with open(args.sections_log_file, 'a', encoding='utf-8') as slog:
                                slog.write(f"{base_doc_name} : {sections}\n")
                            print(f"  [Info] Logged identified sections to '{args.sections_log_file}'. Sections: {sections}", file=sys.stderr)
                        except IOError as e:
                            print(f"  [Warning] Could not write to sections log file '{args.sections_log_file}': {e}", file=sys.stderr)
                        sys.stderr.flush()
                    elif "submitter_name" not in analysis_data_dict or analysis_data_dict["submitter_name"] != "Analysis Error": 
                        print(f"  [Info] No sections identified by LLM for {file_path_in_tar} to log.", file=sys.stderr)
                        sys.stderr.flush()


        markdown_row = format_as_markdown_row(file_path_in_tar, analysis_data_dict)
        print(markdown_row)
        sys.stdout.flush()

        document_processing_end_time = time.monotonic()
        time_spent_this_document = document_processing_end_time - document_processing_start_time
        
        wait_needed = target_document_cycle_seconds - time_spent_this_document
        
        if wait_needed > 0:
            print(f"  [Info] Document processing took {time_spent_this_document:.2f}s. Waiting for {wait_needed:.2f}s to meet {target_document_cycle_seconds}s/doc target rate...", file=sys.stderr)
            sys.stderr.flush()
            time.sleep(wait_needed)
        else:
            print(f"  [Info] Document processing took {time_spent_this_document:.2f}s (target {target_document_cycle_seconds}s/doc). No additional wait needed.", file=sys.stderr)
            sys.stderr.flush()

    print("\n[Info] --- End of Analysis ---", file=sys.stderr)
    sys.stderr.flush()
