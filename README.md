The Earnings Transcript Summary API is a Flask-based web application designed to process PDF earnings transcripts or json payload transcripts. The summaries are generated using Google's Generative AI model (gemini-1.5-flash). It extracts text from uploaded PDFs, preprocesses the content, and generates structured summaries categorized into:

1. Financial Performance
2. Market Dynamics
3. Expansion Plans
4. Environmental Risks
5. Regulatory or Policy Changes

# Features
* PDF Upload: Users can upload PDF files containing earnings transcripts.
* Text Extraction: Extracts text from uploaded PDFs using PyMuPDF (fitz).
* Preprocessing: Cleans and tokenizes text, removing unwanted characters and limiting token count.
* Summarization: Generates structured summaries across predefined categories using a Generative AI model.
* API Endpoint: Provides a POST endpoint to accept JSON payloads for transcript summarization.
* Caching: Implements caching for improved performance on summary retrieval.

main.py file contains all the endpoints while utils.py contains all the functions.

## Prerequisites
* Python 3.7 or higher
* Google Generative AI API Access: Ensure you have a valid API key.
* Replit Account (Optional): For deploying and testing the application on Replit.

### Installation
Clone the Repository: git clone https://github.com/Wiz-2/QuizAI.git

### Create a Virtual Environment:

* python3 -m venv venv
* On Linux: source venv/bin/activate
* On Windows: venv\Scripts\activate

### Install Dependencies:

* pip install -r requirements.txt

#### Set Environment Variables:

* API_KEY: Your Google Generative AI API key.

* Locally:
Create a .env file in the project root:
API_KEY=your_google_api_key
Then, load them in your application.

Ensure API Key Validity:
Verify that your API_KEY has the necessary permissions to access Google's Generative AI services.

# Running the Application
Activate the Virtual Environment:
*source venv/bin/activate  # On Windows: venv\Scripts\activate
* Run the Flask App: python main.py

# API Endpoints
## GET '/'
Description: Renders an HTML form for users to upload a PDF file and enter the company name.
Access URL: [https://quizai-ls9l.onrender.com/]

### Request:
* Method: GET
* URL: /

### Response:
* Status Code: 200 OK
* Body: HTML form

## POST '/upload-pdf'
Description: Handles PDF file uploads, extracts text, stores data in the session, and redirects to the /summary endpoint. It is automatically called when clicking on upload button, which would ultimately redirect to summary endpoint.

### Request:
* Method: POST
* URL: /upload-pdf
* Form Data: textInput (string): Name of the company. file (file): PDF file to upload.

### Response:
* Status Code: 302 Found (Redirect)
* Body: Redirects to /summary
* Errors: 400 Bad Request if file part is missing, no file is selected, or invalid file format.

## GET '/summary'
Description: Retrieves company name and transcript text from the session, generates a summary, and returns it as JSON.

### Request:
* Method: GET
* URL: /summary

### Response:
* Status Code: 200 OK
* Body: JSON object containing summaries.
* Example Response:
{
    "company_name": "Name_of_the_company",
    "environmental_risks": "The text does not contain any information on environmental issues, sustainability, or ESG concerns.",
    "expansion_plans": "Paytm is focused on increasing the monetization of its soundboxes through increased subscription revenues and reactivation efforts. The company is also looking to expand its financial services offerings, with secured loan products expected to contribute meaningfully in the coming quarters. The company is awaiting regulatory approvals to expand its UPI customer base and is committed to investing in its payments business, both for consumer and merchant segments.",
    "financial_performance": "Paytm reported strong financial results for Q2FY25, exceeding previous profitability guidance. The company achieved a significant reduction in cost base, particularly employee costs, leading to a contribution margin of near 55% without UPI incentives. The DLG model, implemented for merchant loans, has not impacted overall profit margins, as the upfront DLG cost is offset by higher collection revenue over the life of the loan. The company is expecting this cost structure to remain stable, and further improvements are anticipated in operating costs.",
    "market_dynamics": "Paytm highlighted the increasing adoption of DLG models in the lending market, driven by regulatory and market practices. This model, while not preferred by all lending partners, has positively impacted the company's ability to secure increased capital for loan disbursement. The company sees a significant opportunity to expand its personal loan distribution business by adding new lending partners and increasing penetration in the market, particularly in tier 2 and 3 cities.",
    "regulatory_or_policy_changes": "Paytm is awaiting regulatory approvals from NPCI to onboard new customers for its UPI platform. The company is also awaiting clarity from the RBI on its payments bank operations. The company is in compliance with all relevant regulatory requirements for its lending business, particularly concerning DLG models and their impact on credit loss coverage."
}
*Errors:400 Bad Request if required data is missing. 500 Internal Server Error if summary generation fails.

## POST '/earnings_transcript_summary'
Description: Accepts a JSON payload containing company_name and transcript_text, generates summaries, and returns them as JSON.
**You can use the below URL with either curl or POSTMAN to access the resource with proper body.
Access URL: [https://quizai-ls9l.onrender.com/earnings_transcript_summary]**

### Request:
* Method: POST
* URL: /earnings_transcript_summary
* Headers:Content-Type: application/json
* Body:{
  "company_name": "Name_of_the_company",
  "transcript_text": "Your transcript text here..."}

### Response:
* Status Code: 200 OK
* Body: JSON object containing summaries.
* Example Response:{
    "company_name": "Paytm",
    "environmental_risks": "The text does not contain any information on environmental issues, sustainability, or ESG concerns.",
    "expansion_plans": "Paytm is focused on increasing the monetization of its soundboxes through increased subscription revenues and reactivation efforts. The company is also looking to expand its financial services offerings, with secured loan products expected to contribute meaningfully in the coming quarters. The company is awaiting regulatory approvals to expand its UPI customer base and is committed to investing in its payments business, both for consumer and merchant segments.",
    "financial_performance": "Paytm reported strong financial results for Q2FY25, exceeding previous profitability guidance. The company achieved a significant reduction in cost base, particularly employee costs, leading to a contribution margin of near 55% without UPI incentives. The DLG model, implemented for merchant loans, has not impacted overall profit margins, as the upfront DLG cost is offset by higher collection revenue over the life of the loan. The company is expecting this cost structure to remain stable, and further improvements are anticipated in operating costs.",
    "market_dynamics": "Paytm highlighted the increasing adoption of DLG models in the lending market, driven by regulatory and market practices. This model, while not preferred by all lending partners, has positively impacted the company's ability to secure increased capital for loan disbursement. The company sees a significant opportunity to expand its personal loan distribution business by adding new lending partners and increasing penetration in the market, particularly in tier 2 and 3 cities.",
    "regulatory_or_policy_changes": "Paytm is awaiting regulatory approvals from NPCI to onboard new customers for its UPI platform. The company is also awaiting clarity from the RBI on its payments bank operations. The company is in compliance with all relevant regulatory requirements for its lending business, particularly concerning DLG models and their impact on credit loss coverage."
}
* Errors: 400 Bad Request for invalid input formats or missing fields. 500 Internal Server Error if summary generation fails.

# Functions
## extract_text_from_pdf(pdf_file)
Description: Extracts text from an uploaded PDF file using PyMuPDF and preprocesses it.

### Parameters: 
* pdf_file (FileStorage): The uploaded PDF file.

### Returns:
* str: The extracted and preprocessed text from the PDF. Returns an empty string if extraction fails.

## preprocess_text(text, max_tokens=20000)
Description: Preprocesses the input text by removing asterisks, cleaning whitespace, tokenizing, truncating to a maximum token limit, and decoding back to text.

### Parameters:
* text (str): The raw input text to preprocess.
* max_tokens (int, optional): The maximum number of tokens allowed in the output. Defaults to 20000.

### Returns:
* str: The preprocessed text.

## generate_summary(company_name, transcript_text, prompt_suffix="Provide the summaries for each category in the form of paragraphs")
Description:Generates a summarized structure from the provided transcript text using a Generative AI model.

### Parameters:
* company_name (str): Name of the company.
* transcript_text (str): Transcript text to summarize.
* prompt_suffix (str, optional): Additional instructions for the AI model. Defaults to "Provide the summaries for each category in the form of paragraphs".

### Returns:
* dict: A dictionary containing the company name and summaries for each category.
* Returns an error dictionary if summary generation fails.

# Error Handling
The API provides clear error messages with appropriate HTTP status codes:

## 400 Bad Request:
Missing or invalid input fields.
Invalid input format.
Transcript text exceeding token limits.
## 500 Internal Server Error:

Failures in summary generation.
Unexpected server-side errors.
Example Error Response:
{
  "error": "Missing or invalid 'company_name' field."
}
