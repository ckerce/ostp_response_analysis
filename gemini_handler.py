# -- ./gemini_handler.py
import google.generativeai as genai
import time
import logging
import json
import re # For parsing JSON from LLM response

# Basic Token Bucket Implementation (for demonstration)
class TokenBucket:
    """
    A simple token bucket algorithm implementation for rate limiting.
    """
    def __init__(self, tokens, fill_rate_per_second):
        """
        Initializes the TokenBucket.

        Args:
            tokens (float): The maximum number of tokens the bucket can hold.
            fill_rate_per_second (float): The rate at which tokens are added to the bucket per second.
        """
        self.capacity = float(tokens)
        self._tokens = float(tokens)  # Current number of tokens in the bucket
        self.fill_rate = float(fill_rate_per_second)
        self.last_consumption_time = time.monotonic() # Last time tokens were refilled/checked
        self.logger = logging.getLogger(__name__) # Logger for this class

    def _refill(self):
        """
        Refills the bucket with new tokens based on the fill rate and elapsed time.
        """
        now = time.monotonic()
        delta_time = now - self.last_consumption_time
        # Add new tokens, ensuring it doesn't exceed capacity
        self._tokens = min(self.capacity, self._tokens + delta_time * self.fill_rate)
        self.last_consumption_time = now

    def consume(self, tokens_to_consume):
        """
        Attempts to consume a specified number of tokens from the bucket.

        Args:
            tokens_to_consume (float): The number of tokens to consume.

        Returns:
            bool: True if tokens were successfully consumed, False otherwise.
        """
        self._refill() # Ensure bucket is up-to-date
        if self._tokens >= tokens_to_consume:
            self._tokens -= tokens_to_consume
            # self.logger.debug(f"Consumed {tokens_to_consume}. Tokens left: {self._tokens:.2f}")
            return True
        # self.logger.debug(f"Not enough tokens to consume {tokens_to_consume}. Tokens available: {self._tokens:.2f}")
        return False

    def get_wait_time(self, tokens_to_consume):
        """
        Calculates the time to wait until the specified number of tokens can be consumed.

        Args:
            tokens_to_consume (float): The number of tokens desired.

        Returns:
            float: The time in seconds to wait. Returns 0 if tokens are immediately available.
                   Returns float('inf') if tokens can never be consumed (e.g., fill_rate is 0).
        """
        self._refill() # Ensure bucket is up-to-date
        if self._tokens >= tokens_to_consume:
            return 0
        
        needed_tokens = tokens_to_consume - self._tokens
        if self.fill_rate <= 0: # Avoid division by zero if fill rate is non-positive
            return float('inf') # Cannot fulfill if no fill rate
        return needed_tokens / self.fill_rate

class GeminiAPIHandler:
    """
    Handles interactions with the Google Gemini API, including rate limiting,
    token tracking, and retries.
    """
    def __init__(self, api_key, model_name, rpm_limit=15, input_tpm_limit=1000000, output_tpm_limit=32000, max_retries=3):
        """
        Initializes the GeminiAPIHandler.

        Args:
            api_key (str): Your Google API Key for the Generative Language API.
            model_name (str): The Gemini model name (e.g., 'gemini-2.0-flash').
            rpm_limit (int): Requests Per Minute limit.
            input_tpm_limit (int): Input Tokens Per Minute limit.
            output_tpm_limit (int): Output Tokens Per Minute limit.
            max_retries (int): Maximum number of retries for API calls on specific errors.
        """
        self.logger = logging.getLogger(__name__)
        try:
            # Configure the genai library with the API key.
            # This should ideally be done once globally if multiple handlers are not needed,
            # but is placed here for encapsulation if the handler is a standalone utility.
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.logger.info(f"GeminiAPIHandler initialized with model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize GenerativeModel '{model_name}': {e}")
            raise # Re-raise exception to signal critical initialization failure

        self.model_name = model_name
        self.max_retries = max_retries

        # Rate Limiters: Convert per-minute limits to per-second fill rates for TokenBuckets
        self.rpm_limiter = TokenBucket(rpm_limit, rpm_limit / 60.0)
        self.input_tpm_limiter = TokenBucket(input_tpm_limit, input_tpm_limit / 60.0)
        self.output_tpm_limiter = TokenBucket(output_tpm_limit, output_tpm_limit / 60.0)

        # Usage Counters
        self.total_requests_made = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_api_errors = 0

    def _count_tokens(self, text_content):
        """Counts tokens for the given text content using the configured model."""
        try:
            # Use the model's count_tokens method for accuracy
            count_response = self.model.count_tokens(text_content)
            return count_response.total_tokens
        except Exception as e:
            self.logger.warning(f"Could not count tokens via API for model {self.model_name}: {e}. Estimating.")
            # Basic estimation as a fallback: split by space and multiply by a factor.
            # This is highly inaccurate and should be used cautiously.
            return len(str(text_content).split()) * 2 # Rough estimate

    def _wait_for_limits(self, input_tokens, estimated_output_tokens_for_tpm_check):
        """Checks and waits if necessary for RPM and TPM limits before making an API call."""
        while True:
            # Calculate wait time for each limiter
            rpm_wait = self.rpm_limiter.get_wait_time(1) # 1 request
            input_tpm_wait = self.input_tpm_limiter.get_wait_time(input_tokens)
            # Pre-flight check for output tokens (using an estimate)
            output_tpm_wait = self.output_tpm_limiter.get_wait_time(estimated_output_tokens_for_tpm_check)

            max_wait_time = max(rpm_wait, input_tpm_wait, output_tpm_wait)

            if max_wait_time > 0:
                self.logger.info(
                    f"Rate limit approached. Waiting {max_wait_time:.2f}s. "
                    f"(RPM: {rpm_wait:.2f}s, InTPM: {input_tpm_wait:.2f}s, EstOutTPM: {output_tpm_wait:.2f}s)"
                )
                time.sleep(max_wait_time)
            else:
                # Attempt to consume tokens if no wait time is calculated.
                # Order of consumption: RPM, then Input TPM. Output TPM is consumed after response.
                if self.rpm_limiter.consume(1):
                    if self.input_tpm_limiter.consume(input_tokens):
                        # Successfully consumed RPM and Input TPM tokens.
                        # Output TPM was only checked for estimated availability.
                        break  # Exit loop, ready for API call
                    else:
                        # Failed to consume input tokens; refund RPM token and retry wait.
                        self.rpm_limiter._tokens = min(self.rpm_limiter.capacity, self.rpm_limiter._tokens + 1)
                        self.logger.debug("Input TPM consumption failed after RPM success; retrying wait.")
                else:
                    self.logger.debug("RPM consumption failed; retrying wait.")
            
            # Add a small safety sleep if we had to wait, to prevent tight looping on edge cases
            if max_wait_time > 0:
                 time.sleep(0.1) # Brief pause before re-checking limits

    def execute_analysis(self, prompt_text):
        """
        Generates content using the Gemini API with rate limiting, token tracking, and retries.

        Args:
            prompt_text (str): The complete prompt to send to the API.

        Returns:
            dict: The parsed JSON response from the API, or a default error structure if all retries fail.
        """
        input_tokens = self._count_tokens(prompt_text)
        
        # Estimate output tokens for pre-flight output TPM check.
        # This is a rough guess. A more sophisticated model might look at prompt length, task type etc.
        # For now, let's assume output might be up to 2x input, capped at a reasonable max (e.g. 2048 for Flash models)
        estimated_output_tokens = min(input_tokens * 2, 2048) # Example cap

        current_retries = self.max_retries
        
        # Default error structure, matching the expected output format
        default_error_output = {
            "submitter_name": "API Handler Error",
            "inferred_submitter_type": "API Handler Error",
            "mission_interest_summary": "Failed to get analysis after multiple retries.",
            "key_concerns": [],
            "policy_recommendations": [],
            "identified_sections": []
        }
        last_response_obj = None # Store the last response object for debugging if needed

        while current_retries > 0:
            self._wait_for_limits(input_tokens, estimated_output_tokens) # Handles waiting

            api_call_start_time = time.monotonic()
            try:
                self.logger.debug(f"Attempting API call. Input tokens: {input_tokens}. Retries left: {current_retries}")
                last_response_obj = self.model.generate_content(prompt_text) # Make the API call
                api_call_duration = time.monotonic() - api_call_start_time
                
                actual_output_tokens = 0
                if hasattr(last_response_obj, 'usage_metadata') and last_response_obj.usage_metadata:
                    actual_output_tokens = last_response_obj.usage_metadata.candidates_token_count or 0
                    # Consume actual output tokens from the output TPM bucket
                    if not self.output_tpm_limiter.consume(actual_output_tokens):
                        self.logger.warning(
                            f"Could not consume actual output tokens ({actual_output_tokens}) "
                            f"from output TPM bucket. Bucket may be exhausted or limit too low."
                        )
                else:
                    self.logger.warning("No usage_metadata in response or it's empty. Output tokens unknown/unaccounted for output TPM.")

                # Update counters for successful call
                self.total_requests_made += 1
                self.total_input_tokens += input_tokens
                self.total_output_tokens += actual_output_tokens # Add actual output tokens
                
                self.logger.info(
                    f"API call successful. Duration: {api_call_duration:.2f}s. "
                    f"Input Tokens: {input_tokens}. Output Tokens: {actual_output_tokens}."
                )

                # Parse the JSON from response_obj.text
                response_text_cleaned = last_response_obj.text.strip()
                # Remove markdown code block fences if present
                if response_text_cleaned.startswith("```json"):
                    response_text_cleaned = re.sub(r"^```json\s*|\s*```$", "", response_text_cleaned, flags=re.DOTALL)
                
                parsed_json = json.loads(response_text_cleaned.strip())
                return parsed_json # Success

            except Exception as e:
                api_call_duration = time.monotonic() - api_call_start_time
                self.total_api_errors += 1
                self.logger.error(
                    f"API call failed (attempt {self.max_retries - current_retries + 1}/{self.max_retries}). "
                    f"Error: {e}. Duration: {api_call_duration:.2f}s"
                )
                
                current_retries -= 1
                if current_retries <= 0:
                    self.logger.error("Max retries reached. Giving up on this request.")
                    return default_error_output # All retries failed

                # Handle specific errors for retry logic
                error_str = str(e)
                error_type_name = type(e).__name__
                
                # Check for rate limit errors (429) or server-side issues (500, 503)
                # google-generativeai might raise google.api_core.exceptions for these.
                if "429" in error_str or "ResourceExhausted" in error_type_name:
                    backoff_time = (2 ** (self.max_retries - current_retries -1)) # Exponential backoff: 1s, 2s, 4s...
                    self.logger.warning(f"Rate limit error detected. Waiting {backoff_time}s before retry...")
                    time.sleep(backoff_time)
                elif "500" in error_str or "503" in error_str or "ServiceUnavailable" in error_type_name or "InternalServerError" in error_type_name:
                    backoff_time = (2 ** (self.max_retries - current_retries -1)) + 1 # Slightly longer for server errors
                    self.logger.warning(f"Server error detected. Waiting {backoff_time}s before retry...")
                    time.sleep(backoff_time)
                else:
                    # For other, potentially non-transient errors, give up immediately
                    self.logger.error(f"Non-retryable or unknown error type encountered: {error_type_name}. Giving up on this request.")
                    return default_error_output
        
        # This line should ideally not be reached if loop logic is correct
        self.logger.error("Exited retry loop unexpectedly. Returning default error.")
        return default_error_output

    def get_usage_summary(self):
        """
        Returns a summary of API usage handled by this instance.
        """
        return {
            "model_name": self.model_name,
            "total_requests_made": self.total_requests_made,
            "total_input_tokens_processed": self.total_input_tokens,
            "total_output_tokens_generated": self.total_output_tokens,
            "total_api_errors": self.total_api_errors
        }
