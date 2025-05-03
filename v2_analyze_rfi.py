#!/usr/bin/env python3
import tarfile
import ollama # Using the official ollama library
import io
import re
import os
import argparse
import sys
import json

# --- Configuration ---
# Defaults - User should adjust these if needed or use command-line args
DEFAULT_TAR_FILE_PATH = 'ostp_responses.tar' # Assumes tar file is in the same directory
DEFAULT_OLLAMA_URL = 'http://172.19.224.1:11434' # Default local Ollama API endpoint
DEFAULT_MODEL_NAME = 'gemma3:27B-it-qat' # The model the user specified
#DEFAULT_MODEL_NAME = 'gemma3:12B-it-qat' # The model the user specified
TARGET_DIR_IN_TAR = 'ostp_md/' # Directory inside the tar file containing .md files
MAX_TEXT_LENGTH = 40000 # Max characters to send to LLM (adjust based on model/memory)

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
            for member in all_members:
                # Ensure the member is within the target directory and is a file ending in .md
                # Use os.path.normpath to handle potential mixed separators
                normalized_member_name = os.path.normpath(member.name)
                normalized_target_dir = os.path.normpath(target_dir)
                if normalized_member_name.startswith(normalized_target_dir) and \
                   normalized_member_name.lower().endswith('.md') and \
                   member.isfile():
                    md_files.append(member.name) # Store original name with original separators
        # Sort for consistent indexing
        md_files.sort()
        print(f"[Info] Found {len(md_files)} '.md' files in '{target_dir}'.", file=sys.stderr)
        return md_files
    except FileNotFoundError:
        print(f"[Error] Tar file not found at '{tar_path}'. Please check the path.", file=sys.stderr)
        return None
    except tarfile.ReadError as e:
         print(f"[Error] Could not read tar file '{tar_path}'. It might be corrupted or not a tar file: {e}", file=sys.stderr)
         return None
    except Exception as e:
        print(f"[Error] An unexpected error occurred while reading tar file: {e}", file=sys.stderr)
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
                # Read content and decode assuming UTF-8, ignore errors for robustness
                content = file_obj.read().decode('utf-8', errors='ignore')
                return content
            else:
                print(f"[Error] Could not get file object for '{file_path_in_tar}' within the tar.", file=sys.stderr)
                return None
    except KeyError:
        print(f"[Error] File not found in tar archive: '{file_path_in_tar}'", file=sys.stderr)
        return None
    except FileNotFoundError:
         print(f"[Error] Tar file not found at '{tar_path}'.", file=sys.stderr)
         return None
    except Exception as e:
        print(f"[Error] An unexpected error occurred extracting file '{file_path_in_tar}': {e}", file=sys.stderr)
        return None

def clean_text(text):
    """
    Basic text cleaning: removes common headers/footers and excessive newlines.
    Args:
        text (str): The raw text content.
    Returns:
        str: The cleaned text.
    """
    cleaned_text = text
    if not text:
        return ""

    # Remove excessive blank lines (more than two consecutive newlines become two)
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    return cleaned_text.strip()

def analyze_document_structure(ollama_client, model_name, text_content, filename):
    """
    First pass: Analyze the document structure to identify major sections.
    Args:
        ollama_client: An initialized ollama.Client instance.
        model_name (str): The name of the Ollama model to use.
        text_content (str): The cleaned text content of the RFI response.
        filename (str): The original filename (used in the prompt).
    Returns:
        str: JSON-formatted string containing the document's section structure.
    """
    # Truncate text if it exceeds the maximum length
    truncated_indicator = "..." if len(text_content) > MAX_TEXT_LENGTH else ""
    content_to_send = text_content[:MAX_TEXT_LENGTH]
    
    structure_prompt = f"""
Analyze the structure of the following RFI response text, submitted under the filename '{filename}'.
The response is to an OSTP RFI on removing barriers to American leadership in AI.

**RFI Response Text (first {MAX_TEXT_LENGTH} characters):**
```markdown
{content_to_send}{truncated_indicator}
```

**Instructions:**
Identify the major sections and structure of this document. Ignore minor formatting elements.
Format your response as a JSON object with the following structure:
```json
{{
  "has_clear_sections": true/false,
  "sections": [
    {{
      "section_id": "1",
      "section_name": "Section name or identifier",
      "section_description": "Brief description of what this section covers"
    }},
    ...
  ]
}}
```

If the document does not have clear sections, set "has_clear_sections" to false and provide an empty array for "sections".
Your response should only contain the JSON object, nothing else.
"""
    
    try:
        print(f" [Info]  Sending structure analysis request to Ollama ({model_name})...", file=sys.stderr)
        response = ollama_client.chat(model=model_name, messages=[
            {'role': 'user', 'content': structure_prompt}
        ])
        print(" [Info]  Received structure analysis from Ollama.", file=sys.stderr)
        
        if 'message' in response and 'content' in response['message']:
            structure_content = response['message']['content']
            
            # Extract JSON from the response (in case there's markdown formatting)
            json_match = re.search(r'```json\s*(.*?)\s*```', structure_content, re.DOTALL)
            if json_match:
                structure_content = json_match.group(1)
            
            # Attempt to parse and validate the JSON
            try:
                structure_data = json.loads(structure_content)
                # Validate expected structure
                if not isinstance(structure_data, dict) or 'has_clear_sections' not in structure_data:
                    print("  [Warning] Structure analysis returned invalid JSON structure.", file=sys.stderr)
                    # Create a default structure
                    structure_data = {"has_clear_sections": False, "sections": []}
                return json.dumps(structure_data)
            except json.JSONDecodeError:
                print("  [Warning] Could not parse structure analysis as valid JSON.", file=sys.stderr)
                # Return a default structure
                return json.dumps({"has_clear_sections": False, "sections": []})
        else:
            print(f"  [Warning] Unexpected response structure from Ollama: {response}", file=sys.stderr)
            return json.dumps({"has_clear_sections": False, "sections": []})
            
    except Exception as e:
        print(f"  [Error] Error during structure analysis: {e}", file=sys.stderr)
        return json.dumps({"has_clear_sections": False, "sections": []})

def analyze_with_ollama(ollama_client, model_name, text_content, filename, structure_data):
    """
    Sends text to a running Ollama server and requests structured analysis.
    Now enhanced with document structure awareness.
    Args:
        ollama_client: An initialized ollama.Client instance.
        model_name (str): The name of the Ollama model to use.
        text_content (str): The cleaned text content of the RFI response.
        filename (str): The original filename (used in the prompt).
        structure_data (str): JSON string containing the document's structure analysis.
    Returns:
        str: The analysis result from Ollama, or an error message string.
    """
    # Truncate text if it exceeds the maximum length
    truncated_indicator = "..." if len(text_content) > MAX_TEXT_LENGTH else ""
    content_to_send = text_content[:MAX_TEXT_LENGTH]
    
    # Parse the structure data
    try:
        structure = json.loads(structure_data)
    except json.JSONDecodeError:
        structure = {"has_clear_sections": False, "sections": []}
    
    # Build the structure info to include in the prompt
    structure_info = ""
    if structure["has_clear_sections"] and structure["sections"]:
        structure_info = "The document has the following structure:\n\n"
        for section in structure["sections"]:
            structure_info += f"- Section {section['section_id']}: {section['section_name']} - {section['section_description']}\n"
    else:
        structure_info = "The document does not have a clear section structure."
    
    prompt = f"""
Analyze the following RFI response text, submitted under the filename '{filename}'.
The response is to an OSTP RFI on removing barriers to American leadership in AI.
Extract the requested information precisely based *only* on the text provided.

**Document Structure:**
{structure_info}

**RFI Response Text (first {MAX_TEXT_LENGTH} characters):**
```markdown
{content_to_send}{truncated_indicator}
```
**Instructions:**
Identify and extract the following information. Present the output using the exact headings and bullet points as shown below. If information is not clearly stated or cannot be inferred from the text, explicitly write "Not Stated" or "Not Applicable".

When listing concerns and recommendations, indicate which section they come from if the document has clear sections.

**Submitter Name:** [Extract submitter's name from filename or text, e.g., Georgia Tech, Verizon, Anonymous, Specific Person]
**Inferred Submitter Type:** [Categorize based on text content and name: e.g., Academia, Industry-Large, Industry-Startup, Industry-Association, Professional-Society, Advocacy/Think Tank, Government, Individual, Anonymous, Unclear]
**Mission/Interest Summary:** [Provide a 1-2 sentence summary of the submitter's primary mission, focus, or interest related to AI policy as evident in the text.]
**Key Concerns:**
"""

    # Add section-specific format for concerns if document has clear sections
    if structure["has_clear_sections"] and structure["sections"]:
        for section in structure["sections"]:
            prompt += f"* [Section {section['section_id']}: {section['section_name']}] [List main concerns from this section. If none, write \"None stated for this section.\"]\n"
    else:
        prompt += "* [List the main concerns, barriers, or risks related to AI mentioned in the text. Use a new bullet for each distinct concern. Write \"None Stated\" if applicable.]\n"

    prompt += "**Policy Recommendations:**\n"

    # Add section-specific format for recommendations if document has clear sections
    if structure["has_clear_sections"] and structure["sections"]:
        for section in structure["sections"]:
            prompt += f"* [Section {section['section_id']}: {section['section_name']}] [List main recommendations from this section. If none, write \"None stated for this section.\"]\n"
    else:
        prompt += "* [List the main policy recommendations, suggestions, or actions proposed in the text. Use a new bullet for each distinct recommendation. Write \"None Stated\" if applicable.]\n"

    prompt += "\nBegin your response now with '**Submitter Name:**'."

    try:
        print(f" [Info]  Sending analysis request to Ollama ({model_name})...", file=sys.stderr)
        response = ollama_client.chat(model=model_name, messages=[
            {'role': 'user', 'content': prompt}
        ])
        print(" [Info]  Received response from Ollama.", file=sys.stderr)
        # Check if the response structure is as expected
        if 'message' in response and 'content' in response['message']:
             return response['message']['content']
        else:
            print(f"  [Warning] Unexpected response structure from Ollama: {response}", file=sys.stderr)
            return "Error during analysis: Unexpected response structure from Ollama."

    except ollama.ResponseError as e:
         print(f"  [Error] Ollama Response Error: Status {e.status_code}, Details: {e.error}", file=sys.stderr)
         return f"Error during analysis: Ollama Response Error {e.status_code}"
    except Exception as e:
        # Catch potential connection errors or other issues
        print(f"  [Error] Could not communicate with Ollama server at {ollama_client._host}: {e}", file=sys.stderr)
        print("  Please ensure the Ollama server is running, accessible, and the model is available.", file=sys.stderr)
        return "Error during analysis: Could not contact Ollama."

def format_as_markdown_row(filename, analysis_result, structure_data):
    """
    Formats the analysis result string (from Ollama) into a Markdown table row.
    Now enhanced to handle section-based analysis.
    Args:
        filename (str): The original filename of the response.
        analysis_result (str): The raw string output from the analyze_with_ollama function.
        structure_data (str): JSON string containing the document's structure analysis.
    Returns:
        str: A formatted Markdown table row string.
    """
    # Default values in case of parsing errors
    data = {
        "Submitter Name": "Parse Error",
        "Inferred Submitter Type": "Parse Error",
        "Mission/Interest Summary": "Parse Error",
        "Key Concerns": "- Parse Error",
        "Policy Recommendations": "- Parse Error"
    }

    # Check if the analysis result itself indicates an error
    if "Error during analysis" in analysis_result:
        data["Submitter Name"] = "Analysis Failed"
        # Escape pipe for table, show error in type field
        data["Inferred Submitter Type"] = analysis_result.replace('|', '\\|')
        data["Mission/Interest Summary"] = "N/A"
        data["Key Concerns"] = "- N/A"
        data["Policy Recommendations"] = "- N/A"

    else:
        # Attempt to parse the expected structure from the LLM response
        current_key = None
        concerns_list = []
        recommendations_list = []
        summary_lines = []

        lines = analysis_result.strip().split('\n')
        for line in lines:
            line_strip = line.strip()

            # Function to extract value and strip markdown formatting
            def extract_value(line_content):
                value = line_content.split(":", 1)[1].strip() if ':' in line_content else line_content
                # Strip leading/trailing asterisks (bold/italic)
                return value.strip('*').strip()

            # Use startswith for robustness against extra whitespace
            if line_strip.startswith("**Submitter Name:**"):
                data["Submitter Name"] = extract_value(line_strip)
                current_key = None # Reset key
            elif line_strip.startswith("**Inferred Submitter Type:**"):
                data["Inferred Submitter Type"] = extract_value(line_strip)
                current_key = None
            elif line_strip.startswith("**Mission/Interest Summary:**"):
                summary_part = extract_value(line_strip)
                summary_lines.append(summary_part)
                current_key = "Summary" # Set key to capture multi-line summaries
            elif line_strip.startswith("**Key Concerns:**"):
                current_key = "Concerns"
            elif line_strip.startswith("**Policy Recommendations:**"):
                current_key = "Recommendations"
            # Special handling for section-based bullets
            elif line_strip.startswith("* [Section") and current_key == "Concerns":
                # Match format: "* [Section X: Name] Concern text"
                section_match = re.match(r'\* \[Section (.*?)(?:\: (.*?))?\] (.*)', line_strip)
                if section_match:
                    section_id = section_match.group(1)
                    section_name = section_match.group(2) or ""
                    concern_text = section_match.group(3)
                    
                    # Format with section info
                    if section_name:
                        concerns_list.append(f"[§{section_id}: {section_name}] {concern_text}")
                    else:
                        concerns_list.append(f"[§{section_id}] {concern_text}")
                else:
                    # If regex doesn't match but it starts with * [Section, add it anyway
                    concerns_list.append(line_strip[2:])
            elif line_strip.startswith("* [Section") and current_key == "Recommendations":
                # Match format: "* [Section X: Name] Recommendation text"
                section_match = re.match(r'\* \[Section (.*?)(?:\: (.*?))?\] (.*)', line_strip)
                if section_match:
                    section_id = section_match.group(1)
                    section_name = section_match.group(2) or ""
                    rec_text = section_match.group(3)
                    
                    # Format with section info
                    if section_name:
                        recommendations_list.append(f"[§{section_id}: {section_name}] {rec_text}")
                    else:
                        recommendations_list.append(f"[§{section_id}] {rec_text}")
                else:
                    # If regex doesn't match but it starts with * [Section, add it anyway
                    recommendations_list.append(line_strip[2:])
            # Standard bullet handling
            elif line_strip.startswith("* ") and current_key == "Concerns":
                # Strip potential markdown from list item
                concerns_list.append(line_strip[2:].strip('*').strip())
            elif line_strip.startswith("* ") and current_key == "Recommendations":
                 # Strip potential markdown from list item
                recommendations_list.append(line_strip[2:].strip('*').strip())
            elif current_key == "Summary" and line_strip and not line_strip.startswith("**"): # Continue summary
                 # Strip potential markdown from continuation line
                 summary_lines.append(line_strip.strip('*').strip())
            elif not line_strip.startswith("**"): # If line doesn't start with ** and isn't a list item, reset key
                 current_key = None

        # Join multi-line summary and lists
        data["Mission/Interest Summary"] = " ".join(summary_lines) if summary_lines else "Not Stated"
        
        # Add bullet points back after stripping
        data["Key Concerns"] = "<br>".join(f"- {item}" for item in concerns_list) if concerns_list else "- None Stated"
        data["Policy Recommendations"] = "<br>".join(f"- {item}" for item in recommendations_list) if recommendations_list else "- None Stated"

        # Check if parsing still looks wrong
        if data["Submitter Name"] == "Parse Error" and data["Inferred Submitter Type"] == "Parse Error":
             print(f"  [Warning] Failed to parse LLM output for {filename}. Raw output snippet:", file=sys.stderr)
             snippet_text = analysis_result[:150].replace('|',' ')
             print(f" [Info]  Snippet: {snippet_text}...", file=sys.stderr)
             # Keep defaults indicating parse error

    # Escape pipe characters within the final content for table integrity
    for key in data:
        if isinstance(data[key], str):
            # Ensure escaping happens *after* all processing and joining
            data[key] = data[key].replace('|', '\\|')

    # Extract base filename without path
    base_filename = os.path.basename(filename)

    return f"| {base_filename} | {data['Submitter Name']} | {data['Inferred Submitter Type']} | {data['Mission/Interest Summary']} | {data['Key Concerns']} | {data['Policy Recommendations']} |"

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze RFI responses (.md files) from a tar archive using a local Ollama server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument("indices", metavar="K", type=int, nargs='*', # Changed to '*' to allow zero indices if --list_files is used
                        help="One or more 0-based indices of the files to process within the tar archive (based on sorted list). If omitted and --list_files is not used, processes no files.")
    parser.add_argument("--tarfile", default=DEFAULT_TAR_FILE_PATH,
                        help="Path to the tar archive.")
    parser.add_argument("--ollama_url", default=DEFAULT_OLLAMA_URL,
                        help="URL for the Ollama server.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help="Ollama model name.")
    parser.add_argument("--list_files", action="store_true",
                        help="List all .md files found in the archive's target directory with their indices and exit.")
    parser.add_argument("--target_dir", default=TARGET_DIR_IN_TAR,
                        help="Target directory within the tar archive containing .md files.")
    parser.add_argument("--disable_structure", action="store_true",
                        help="Disable section structure analysis (faster but less detailed).")

    args = parser.parse_args()

    # --- 1. Get list of files ---
    all_files = list_md_files_in_tar(args.tarfile, args.target_dir)
    if all_files is None: # Check if list_md_files_in_tar indicated an error
        sys.exit(1)
    if not all_files:
        print(f"[Warning] No '.md' files found in '{args.target_dir}' within '{args.tarfile}'.", file=sys.stderr)
        # Exit cleanly if no files found, unless --list_files was specified (handled below)
        if not args.list_files:
             sys.exit(0)

    # --- Handle --list_files action ---
    if args.list_files:
        if all_files:
            print("\nFiles found (Index: Path):")
            for i, f in enumerate(all_files):
                print(f"{i}: {f}")
        else:
             print(f"\nNo '.md' files found in '{args.target_dir}' within '{args.tarfile}'.")
        sys.exit(0) # Exit after listing

    # --- If not listing files, ensure indices were provided ---
    if not args.indices:
         print("[Error] No file indices provided for analysis. Use --list_files to see available indices.", file=sys.stderr)
         sys.exit(1)

    # --- 2. Determine files to process based on indices ---
    files_to_process = []
    invalid_indices = []
    max_index = len(all_files) - 1
    for k in args.indices:
        if 0 <= k <= max_index:
            files_to_process.append(all_files[k])
        else:
            invalid_indices.append(k)

    if invalid_indices:
        print(f"\n[Error] Invalid index/indices provided: {invalid_indices}", file=sys.stderr)
        print(f"Valid indices range from 0 to {max_index}.", file=sys.stderr)
        sys.exit(1)

    if not files_to_process:
        # This case should ideally not be reached if args.indices is required when not listing
        print("[Error] No valid file indices resulted in files to process.", file=sys.stderr)
        sys.exit(1)

    # --- 3. Initialize Ollama client ---
    try:
        print(f"\n[Info] Attempting to connect to Ollama at {args.ollama_url}...", file=sys.stderr)
        client = ollama.Client(host=args.ollama_url)
        # Test connection by listing models - provides feedback if server is down
        client.list()
        print(f"[Info] Successfully connected to Ollama.", file=sys.stderr)

    except Exception as e:
        print(f"\n[Error] Failed to connect to Ollama at {args.ollama_url}: {e}", file=sys.stderr)
        print("Please ensure the Ollama server is running, accessible, and the model is available.", file=sys.stderr)
        sys.exit(1)

    # --- 4. Print Markdown Table Header to Standard Output ---
    # Use standard output for the table, keep logs/errors on standard error
    print("| Filename | Submitter Name | Submitter Type | Mission/Interest Summary | Key Concerns | Policy Recommendations |")
    print("|---|---|---|---|---|---|")

    # --- 5. Process selected files ---
    for file_path in files_to_process:
        print(f"\nProcessing: {file_path}...", file=sys.stderr)
        # Extract text
        text = extract_text_from_tar(args.tarfile, file_path)
        if text is None: # Check explicitly for None which indicates an extraction error
            print(f"  Skipping {file_path} due to extraction error.", file=sys.stderr)
            # Optionally print an error row to the table
            print(f"| {os.path.basename(file_path)} | Extraction Error | N/A | N/A | N/A | N/A |")
            continue # Skip to the next file

        # Clean text
        cleaned_text = clean_text(text)
        if not cleaned_text:
             print(f"  Skipping {file_path} as content is empty after cleaning.", file=sys.stderr)
             # Optionally print an empty row
             print(f"| {os.path.basename(file_path)} | Empty Content | N/A | N/A | N/A | N/A |")
             continue
        print(f"  Extracted and cleaned text (length: {len(cleaned_text)} chars).", file=sys.stderr)

        # Analyze document structure if not disabled
        structure_data = json.dumps({"has_clear_sections": False, "sections": []})
        if not args.disable_structure:
            structure_data = analyze_document_structure(client, args.model, cleaned_text, file_path)
            try:
                structure = json.loads(structure_data)
                if structure["has_clear_sections"]:
                    print(f"  Identified {len(structure['sections'])} document sections.", file=sys.stderr)
                else:
                    print(f"  No clear document structure identified.", file=sys.stderr)
            except:
                print(f"  Warning: Could not parse structure data.", file=sys.stderr)

        # Analyze with LLM
        analysis_result = analyze_with_ollama(client, args.model, cleaned_text, file_path, structure_data)

        # Format and print table row to standard output
        markdown_row = format_as_markdown_row(file_path, analysis_result, structure_data)
        print(markdown_row)

    print("\n [Info] --- End of Analysis ---", file=sys.stderr)
