# app.py
from flask import Flask, request, jsonify, render_template_string, redirect, url_for, session
from flask_caching import Cache
import os
import logging
from utils import preprocess_text, extract_text_from_pdf, generate_summary, tokenizer
import google.generativeai as genai

app = Flask(__name__)

session = {}

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Configure Caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Configure Generative AI API
api_key = os.environ.get("API_KEY")
if not api_key:
    raise EnvironmentError("API_KEY is not set in the environment variables.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route('/')
def index():
    return render_template_string('''
        <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Upload PDF Form</title>
            </head>
            <body>
                <h2>Upload PDF File</h2>
                <form action="/upload-pdf" method="post" enctype="multipart/form-data">
                    <label for="textInput">Enter the Company Name</label>
                    <input type="text" id="textInput" name="textInput" required>
                    <label for="file">Select PDF file to upload:</label>
                    <input type="file" id="file" name="file" accept=".pdf" required>
                    <br><br>
                    <button type="submit">Submit</button>
                </form>
            </body>
            </html>
        ''')

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    company_name = request.form.get("textInput")

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.pdf'):
        pdf_text = extract_text_from_pdf(file)

        session['company_name'] = company_name
        session['text'] = pdf_text

        # Redirect to the summary endpoint
        return redirect(url_for('summary'))

    else:
        return jsonify({'error': 'Invalid file format. Only PDF files are allowed.'}), 400

@app.route('/summary')
@cache.cached(timeout=300, query_string=True)
def summary():
    company_name = session.get('company_name')
    transcript_text = session.get('text')

    if not transcript_text or not company_name:
        return jsonify({"error": "Missing required data"}), 400

    # Call the centralized summary function
    output = generate_summary(
            company_name=company_name,
            transcript_text=transcript_text,
            model=model  # Pass the configured model
        )

    if "error" in output:
        return jsonify({"error": output["error"]}), 500

    return jsonify(output), 200

@app.route('/earnings_transcript_summary', methods=['POST'])
@cache.cached(timeout=300, query_string=True)
def earnings_transcript_summary():
    """
    Endpoint to summarize earnings transcript into defined categories.
    """
    try:
        # Ensure the request is in JSON format
        if not request.is_json:
            return jsonify({"error": "Invalid input format. JSON expected."}), 400

        data = request.get_json()

            # Validate input fields
        company_name = data.get("company_name")
        transcript_text = data.get("transcript_text")

        if not company_name or not isinstance(company_name, str):
            return jsonify({"error": "Missing or invalid 'company_name' field."}), 400

        if not transcript_text or not isinstance(transcript_text, str):
            return jsonify({"error": "Missing or invalid 'transcript_text' field."}), 400

        MAX_ALLOWED_TOKENS = 20000
        token_count = len(tokenizer.encode(transcript_text))
        if token_count > MAX_ALLOWED_TOKENS:
            return jsonify({"error": f"'transcript_text' exceeds the maximum allowed length of {MAX_ALLOWED_TOKENS} tokens."}), 400

            # Call the generate_summary function
        output = generate_summary(
            company_name=company_name,
            transcript_text=transcript_text,
            model=model  # Pass the configured model
            )

        if "error" in output:
            return jsonify({"error": output["error"]}), 500

        return jsonify(output), 200

    except Exception as e:
        logging.error(f"An error occurred in /earnings_transcript_summary: {e}", exc_info=True)
        return jsonify({"error": "Internal server error."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
