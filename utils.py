import re
import logging
import fitz
from transformers import GPT2Tokenizer
import google.generativeai as genai

# Initialize the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def preprocess_text(text, max_tokens=20000):
    """
    Preprocess the input text by removing unnecessary characters, cleaning whitespace,
    tokenizing, truncating to a maximum token limit, and decoding back to text.

    Parameters:
        text (str): The raw input text to preprocess.
        max_tokens (int): The maximum number of tokens allowed in the output.

    Returns:
        str: The preprocessed text.
    """
    try:
        #Remove asterisks
        cleaned_text = re.sub(r'\*', '', text)

        # Remove unnecessary newlines and whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Tokenize using GPT-2 tokenizer
        tokens = tokenizer.encode(cleaned_text)

        # Truncate tokens if exceeding max_tokens
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens]
        else:
            truncated_tokens = tokens

        # Decode the tokens back to text
        truncated_text = tokenizer.decode(truncated_tokens)

        return truncated_text

    except Exception as e:
        logging.error(f"Error in preprocess_text: {e}", exc_info=True)
        return ''

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file using PyMuPDF, used when using '/' route to upload the pdf instead of transcript text and preprocesses it.

    Parameters:
        pdf_file (FileStorage): The uploaded PDF file.

    Returns:
        str: The extracted and preprocessed text from the PDF.
    """
    try:
        pdf_reader = fitz.open("pdf", pdf_file.read())
        text = ''
        for page in pdf_reader:
            text += page.get_text()
        pdf_reader.close()
        text = preprocess_text(text)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}", exc_info=True)
        return ''

def generate_summary(company_name, transcript_text, model, prompt_suffix="Provide the summaries for each category in the form of paragraphs"):
    """
    Generates a summarized structure from the provided transcript text.

    Parameters:
        company_name (str): Name of the company.
        transcript_text (str): Transcript text to summarize.
        model: Configured Generative AI model.
        prompt_suffix (str): Additional instructions for the AI model.
        
    Returns:
        dict: A dictionary containing the company name and summaries for each category.
    """
    try:
        # Preprocess the transcript text
        preprocessed_text = preprocess_text(transcript_text, max_tokens=20000)

        # Construct the prompt for summarization
        prompt = (
            """Please summarize the given text into the following categories:
    Financial Performance: Summarize key financial metrics or statements about the company's recent performance.
    Market Dynamics: Summarize any commentary on market trends, demand shifts, competition, etc.
    Expansion Plans: Summarize any information on the company's plans for growth or expansion.
    Environmental Risks: Summarize references to environmental issues, sustainability, or ESG concerns.
    Regulatory or Policy Changes: Summarize any information on recent or upcoming regulatory or policy changes affecting the company.

    Text:""" + preprocessed_text + """

    """ + prompt_suffix
        )

        # Generate the response using the AI model
        response = model.generate_content(prompt)

        if not response or not hasattr(response, 'text'):
            logging.error("Failed to generate summary from the AI model.")
            return {"error": "Failed to generate summary."}

        response_text = response.text

        # Parse the response to extract summaries for each category
        output_dict = {}
        categories = [
            "Financial Performance",
            "Market Dynamics",
            "Expansion Plans",
            "Environmental Risks",
            "Regulatory or Policy Changes"
        ]

        for i, category in enumerate(categories):
            # Find the starting index of the category in the response
            start_idx = response_text.find(category + ":")

            # Determine the ending index, which is either the start of the next category or the end of the text
            if i < len(categories) - 1:
                end_idx = response_text.find(categories[i + 1] + ":", start_idx)
            else:
                end_idx = len(response_text)

            # Extract the summary for the category and strip any unnecessary whitespace
            if start_idx != -1:
                summary = response_text[start_idx + len(category) + 1:end_idx].strip()
                summary = preprocess_text(summary, max_tokens=256)  # Shorten summaries if necessary
                key = category.lower().replace(" ", "_")
                output_dict[key] = summary

        output_dict["company_name"] = company_name

        return output_dict

    except Exception as e:
        logging.error(f"An error occurred in generate_summary: {e}", exc_info=True)
        return {"error": "Internal server error."}
