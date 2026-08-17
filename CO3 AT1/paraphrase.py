import os
import re
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

# ============================================================
# HUGGING FACE PARAPHRASING ASSISTANT
# ============================================================
#
# Model      : Qwen/Qwen2.5-7B-Instruct
# Platform   : Hugging Face Inference API
# Temperature: 0.7
# Max Tokens : 300
#
# The Hugging Face API token is loaded securely from .env
# and is NOT hard-coded in this program.
# ============================================================

# Load environment variables from .env
load_dotenv()

# Read Hugging Face API token
HF_TOKEN = os.getenv("HF_TOKEN")

# Check whether token exists
if not HF_TOKEN:
    print("\nERROR: HF_TOKEN was not found.")
    print("Please create a .env file containing:")
    print("HF_TOKEN=your_token_here")
    exit()

# Model used for paraphrasing
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Generation parameters
TEMPERATURE = 0.7
MAX_TOKENS = 300

# Create Hugging Face inference client
client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto",
    timeout=30
)


# ============================================================
# FUNCTION: CHECK TECHNICAL TERM
# ============================================================

def check_technical_term(text, technical_term):
    """
    Checks whether the required technical term appears
    in the generated paraphrase.
    """

    if not technical_term:
        return True

    return technical_term.lower() in text.lower()


# ============================================================
# FUNCTION: GENERATE PARAPHRASES
# ============================================================

def generate_paraphrases(sentence, technical_term=""):
    """
    Sends the sentence to Hugging Face and generates
    exactly three distinct paraphrases.
    """

    if not sentence.strip():
        print("\nERROR: Input sentence cannot be empty.")
        return []

    # Create a strict instruction for the model
    if technical_term:
        term_instruction = f"""
IMPORTANT:
The technical term "{technical_term}" MUST appear exactly
in all three paraphrases. Do not remove, replace, translate,
or modify this technical term.
"""
    else:
        term_instruction = ""

    prompt = f"""
You are a professional paraphrasing assistant.

Rewrite the following sentence into exactly THREE distinct
paraphrases.

Rules:
1. Preserve the original meaning.
2. Do not add new facts.
3. Do not remove important information.
4. Make each paraphrase grammatically correct.
5. Make the three versions clearly different.
6. Return ONLY the three paraphrases.
7. Use exactly this format:

PARAPHRASE 1: ...
PARAPHRASE 2: ...
PARAPHRASE 3: ...

{term_instruction}

Original sentence:
{sentence}
"""

    try:

        # Send request to Hugging Face
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        # Extract generated text
        result = response.choices[0].message.content.strip()

        # Parse the three paraphrases
        paraphrases = []

        pattern = r"PARAPHRASE\s*[123]\s*:\s*(.*?)(?=\nPARAPHRASE\s*[123]\s*:|$)"

        matches = re.findall(
            pattern,
            result,
            flags=re.IGNORECASE | re.DOTALL
        )

        for match in matches:
            cleaned = match.strip()

            if cleaned:
                paraphrases.append(cleaned)

        # Fallback if the model didn't follow the format
        if len(paraphrases) < 3:

            lines = [
                line.strip()
                for line in result.split("\n")
                if line.strip()
            ]

            cleaned_lines = []

            for line in lines:

                line = re.sub(
                    r"^(PARAPHRASE\s*[123]\s*:\s*)",
                    "",
                    line,
                    flags=re.IGNORECASE
                )

                if line.strip():
                    cleaned_lines.append(line.strip())

            paraphrases = cleaned_lines[:3]

        return paraphrases[:3]

    except HfHubHTTPError as e:

        # Handle HTTP errors returned by Hugging Face
        error_text = str(e)

        if "429" in error_text:
            print("\nERROR: Rate limit or API quota exceeded.")
            print("Please wait and try again later.")

        elif "401" in error_text or "403" in error_text:
            print("\nERROR: Hugging Face authentication failed.")
            print("Check your HF_TOKEN in the .env file.")

        else:
            print("\nERROR: Hugging Face API failure.")
            print(error_text)

        return []

    except TimeoutError:

        # Handle timeout
        print("\nERROR: API request timed out.")
        print("Please check your internet connection and try again.")

        return []

    except Exception as e:

        # Handle any unexpected API/program error
        error_text = str(e)

        if "429" in error_text:
            print("\nERROR: Rate limit or quota exceeded.")

        elif "timeout" in error_text.lower():
            print("\nERROR: API request timed out.")

        else:
            print("\nERROR: Unexpected API failure.")
            print(error_text)

        return []


# ============================================================
# FUNCTION: DISPLAY RESULTS
# ============================================================

def display_results(sentence_number, sentence, paraphrases, technical_term=""):

    print("\n" + "=" * 70)
    print(f"SENTENCE {sentence_number}")
    print("=" * 70)

    print("\nOriginal:")
    print(sentence)

    if not paraphrases:
        print("\nNo paraphrases were generated.")
        return

    print("\nGenerated Paraphrases:")

    for index, paraphrase in enumerate(paraphrases, start=1):

        print(f"\n{index}. {paraphrase}")

    # Technical term verification
    if technical_term:

        print("\n" + "-" * 70)
        print("TECHNICAL TERM CHECK")
        print("-" * 70)

        print(f'Required term: "{technical_term}"')

        for index, paraphrase in enumerate(paraphrases, start=1):

            if check_technical_term(paraphrase, technical_term):
                print(f"Paraphrase {index}: PRESERVED")
            else:
                print(f"Paraphrase {index}: NOT PRESERVED")


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("              HUGGING FACE PARAPHRASING ASSISTANT")
    print("=" * 70)

    print(f"\nModel      : {MODEL_NAME}")
    print("Platform   : Hugging Face Inference API")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Max Tokens : {MAX_TOKENS}")

    print("\nThe program will process THREE sentences.")
    print("For a technical-term example, enter the required")
    print("technical term when requested.")
    print("\n" + "-" * 70)

    results = []

    # --------------------------------------------------------
    # Get three sentences
    # --------------------------------------------------------

    for i in range(1, 4):

        while True:

            sentence = input(
                f"\nEnter sentence {i}: "
            ).strip()

            # Validate input
            if not sentence:
                print("ERROR: Sentence cannot be empty.")
                print("Please enter a valid sentence.")
                continue

            break

        # Ask whether a technical term must be preserved
        technical_term = input(
            "Technical term to preserve (press Enter if none): "
        ).strip()

        print("\nGenerating three paraphrases...")

        paraphrases = generate_paraphrases(
            sentence,
            technical_term
        )

        display_results(
            i,
            sentence,
            paraphrases,
            technical_term
        )

        results.append(
            (sentence, paraphrases, technical_term)
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("                    PROCESS COMPLETED")
    print("=" * 70)

    successful = sum(
        1 for _, paraphrases, _ in results
        if len(paraphrases) >= 3
    )

    print(
        f"\nSuccessfully processed: {successful}/3 sentences"
    )

    print("\nThank you for using the Paraphrasing Assistant.")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()