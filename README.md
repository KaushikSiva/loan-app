🎥 [Watch the Loom video](https://www.loom.com/share/b53c8754367d4d92948d1bdf132424a1)

Document Processing and Embedding System
This repository contains the code for a document processing and embedding system that extracts financial data from documents, generates insights using AI models, and stores the results in a Supabase database. The system is capable of handling various document formats, including PDFs, and performs tasks such as financial analysis, embedding generation, and document retrieval.

Workflow Overview
Ingesting Data:

PDF documents are ingested, containing financial data and other related information.
The system extracts relevant data, such as "Total Monthly Deposits", "Regular Bills", "Outstanding Loans", and more, using both AI models and rule-based logic.
Processing the Data:

The extracted information is processed using AI models, including Groq for generating financial insights and decisions.
The relevant document context is retrieved using embedding-based search to answer user queries.
Storing Results:

The processed results, including financial insights and loan eligibility, are stored in the Supabase database for further analysis and retrieval.
Setup Instructions
Prerequisites
Python 3.8+
Required libraries:
langchain, sentence-transformers, sklearn, supabase, langchain-groq, dotenv, IPython, re, json, numpy
Supabase Account and Key for database operations
Access to necessary AI models, including FINANCIAL_INSIGHTS_MODEL, EMBEDDING_MODEL
Install Dependencies
Clone the repository:

bash
Copy
Edit
git clone <repository_url>
cd <repository_folder>
Install required Python packages:

bash
Copy
Edit
pip install -r requirements.txt
Create a .env file in the root directory with the following values:

bash
Copy
Edit
FINANCIAL_INSIGHTS_PROMPT="Your prompt for financial insights"
JSON_EXTRACTOR_PROMPT="Your prompt for JSON extraction"
EMBEDDING_MODEL="Your embedding model name"
FINANCIAL_INSIGHTS_MODEL="Your financial insights model"
SUPABASE_URL="Your Supabase URL"
SUPABASE_KEY="Your Supabase Key"
Load environment variables in your script:

python
Copy
Edit
from dotenv import load_dotenv
load_dotenv()
Workflow Details
1. Ingesting Data:
The system begins by extracting relevant financial details from PDF documents using the following functions:

generate_answer(): Uses the Groq model to retrieve answers from documents.
retrieve_document_by_type_and_id(): Retrieves relevant documents based on file type and ID.
extract_json_from_input(): Extracts embedded JSON data from documents.
Example:

python
Copy
Edit
json_from_txt = extract_json_from_input(generate_answer("txt", f"{loan_id}.pdf", loan_id))
json_from_img = extract_json_from_input(generate_answer("img_txt", f"{loan_id}.pdf", loan_id))
2. Processing the Data:
After data ingestion, the system processes and merges data from multiple sources into a unified format:

merge_details(): Merges data from various extracted documents.
generate_financial_insights(): Uses AI models to generate financial insights and evaluate loan eligibility.
Example:

python
Copy
Edit
merged_data = merge_details(json_from_txt, json_from_img, json_from_table)
merged_data, think = generate_financial_insights(merged_data)
3. Storing Results:
Once processed, the data is formatted and stored in the Supabase database:

convert_to_loan_analysis_data(): Converts the processed data into the required format.
insert_loan_analysis_data(): Inserts the formatted data into Supabase.
Example:

python
Copy
Edit
loan_analysis_data = convert_to_loan_analysis_data(loan_id, merged_data, think)
insert_loan_analysis_data(loan_analysis_data)
Core Functions
get_embedding()
Generates embeddings for the input text using the specified embedding model.

python
Copy
Edit
def get_embedding(text):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding = model.encode(text)
    return embedding.tolist()
generate_financial_insights()
Generates financial insights and loan eligibility from the merged data.

python
Copy
Edit
def generate_financial_insights(merged_data):
    prompt = FINANCIAL_INSIGHTS_PROMPT
    chat = ChatGroq(model_name=FINANCIAL_INSIGHTS_MODEL)
    messages = [HumanMessage(content=prompt)]
    response = chat(messages).content.strip()
    think = extract_think_content(response)
    insights, eligibility = extract_insights_le(response)
    merged_data["Financial Insights"] = insights
    merged_data["Loan Eligibility"] = eligibility
    return merged_data, think
insert_loan_analysis_data()
Inserts loan analysis data into the Supabase database.

python
Copy
Edit
def insert_loan_analysis_data(loan_analysis_data):
    response = supabase_client.table('loan_analysis').insert({
        "loan_id": loan_analysis_data["loan_id"],
        "manual_loan_approval": loan_analysis_data["manual_loan_approval"],
        "manual_loan_status": loan_analysis_data["manual_loan_status"],
        "ai_decision": loan_analysis_data["ai_decision"],
        "total_monthly_deposits": loan_analysis_data["total_monthly_deposits"],
        "total_monthly_withdrawals": loan_analysis_data["total_monthly_withdrawals"],
        "regular_bills": loan_analysis_data["regular_bills"],
        "outstanding_loans": loan_analysis_data["outstanding_loans"],
        "financial_insights": loan_analysis_data["financial_insights"],
        "ai_think": loan_analysis_data["ai_think"]
    }).execute()
    return response
merge_details()
Merges extracted financial data from multiple sources into a unified structure.

python
Copy
Edit
def merge_details(*details_list):
    merged = {
        "Total Monthly Deposits": 0,
        "Total Monthly Withdrawals": 0,
        "Regular Bills": [],
        "Outstanding Loans": []
    }
    # Code for merging details...
    return merged
Example Use Case
To perform a loan analysis, call the do_loan_analysis() function, passing in the loan_id:

python
Copy
Edit
do_loan_analysis(loan_id)
Expected Output:
The system will output:

Merged data from various document sources.
Generated financial insights for the loan analysis.
Results stored in the Supabase database.
