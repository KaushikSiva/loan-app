🎥 [Watch the Loom video](https://www.loom.com/share/b53c8754367d4d92948d1bdf132424a1)

# Loan Analysis System

This repository contains the code for a Loan Analysis System that processes PDF documents, extracts relevant information, and performs loan eligibility assessments using AI.

## Features

- **PDF Parsing and Chunking**: Utilizes `unstructured` to parse PDF documents, extract text, tables, and images.
- **AI-Powered Information Extraction**: Leverages `LangChain` and `Groq` models for text summarization, table conversion, and image-to-text processing.
- **Embedding Generation**: Uses `SentenceTransformers` to generate embeddings for text chunks, enabling semantic search.
- **Data Storage**: Stores document embeddings and loan analysis results in a Supabase database.
- **Loan Eligibility Assessment**: Performs loan eligibility assessment based on extracted financial insights.
- **Flask API**: Provides a RESTful API for file uploads, loan analysis, and data retrieval.

## Prerequisites

- Python 3.7+
- Docker (for running Supabase locally, if needed)
- Supabase account (or local setup)
- Groq API key
- SentenceTransformers model (specified in `.env`)
- Required Python packages (install using `pip install -r requirements.txt`)

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>

2. **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS and Linux
    venv\Scripts\activate  # On Windows

3. **pip install -r requirements.txt**

4. **Set up environment variables:**
```bash
    GROQ_API_KEY=<your_groq_api_key>
    LANGCHAIN_API_KEY=<your_langchain_api_key>
    LANGCHAIN_TRACING_V2=false # or true
    HF_TOKEN=<your_huggingface_token> # if needed for huggingface models
    SUPABASE_URL=<your_supabase_url>
    SUPABASE_KEY=<your_supabase_key>
    EMBEDDING_MODEL=<your_embedding_model_name> # e.g., all-mpnet-base-v2
    IMAGE_TO_TEXT_MODEL=<your_image_to_text_model_name> # e.g., mixtral-8x7b-32768
    TABLE_TO_TEXT_MODEL=<your_table_to_text_model_name> # e.g., mixtral-8x7b-32768
    TABLE_TO_TEXT_PROMPT="Summarize this table in text format."
    IMAGE_TRANSACTION_SUMMARY_PROMPT="."
    FINANCIAL_INSIGHTS_PROMPT=""
    JSON_EXTRACTOR_PROMPT=""
    FINANCIAL_INSIGHTS_MODEL=<your_financial_insights_model_name>
```
5. **Run Supabase**

6. **python app.py**

***Usage***
1. Upload a PDF file:

Access the /upload route in your browser or use a tool like curl or Postman to upload a PDF file to the /upload-file/<loan_id> endpoint. Replace <loan_id> with a unique loan ID.

2. Trigger Loan Analysis:
Trigger the loan analysis with the loan id.
Access the /loan-analysis/<loan_id> endpoint to retrieve the loan analysis results. Replace <loan_id> with the loan ID.

3. Approve a loan:
Use the /approve-loan endpoint with a POST request to approve or reject a loan.


***File Structure***

```bash
loan-analysis-system/
├── app.py
├── pdf_embedder.py
├── retreiver.py
├── requirements.txt
├── .env
├── templates/
│   ├── index.html
│   └── index2.html
└── /tmp/uploads/ # Folder for uploaded files
```

**Dependencies***

Flask
Supabase Python client
LangChain
Groq Python client
Unstructured
Sentence Transformers
Scikit-learn
Python-dotenv
