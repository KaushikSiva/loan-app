import os
from unstructured.partition.pdf import partition_pdf
import base64
from IPython.display import Image, display
from langchain.schema import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client, Client
from langchain_core.prompts import PromptTemplate
import numpy as np
import json
import re
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from dotenv import load_dotenv



load_dotenv()

FINANCIAL_INSIGHTS_PROMPT  = os.getenv("FINANCIAL_INSIGHTS_PROMPT", "").replace("\\n", "\n")
JSON_EXTRACTOR_PROMPT = os.getenv("JSON_EXTRACTOR_PROMPT", "").replace("\\n", "\n")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
FINANCIAL_INSIGHTS_MODEL = os.getenv("FINANCIAL_INSIGHTS_MODEL")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_embedding(text):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding = model.encode(text)
    return embedding.tolist()

# Function to retrieve relevant document based on query


# Function to convert the data
def convert_to_loan_analysis_data(loan_id, input_data, ai_think):
    loan_analysis_data = {
        "loan_id" : loan_id,
        "manual_loan_approval": False,  # assuming manual approval is False
        "manual_loan_status": "Pending",  # assuming manual status is Pending
        "ai_decision": input_data['Loan Eligibility'],  # AI decision based on eligibility
        "total_monthly_deposits": input_data['Total Monthly Deposits'],
        "total_monthly_withdrawals": input_data['Total Monthly Withdrawals'],
        "regular_bills": input_data['Regular Bills'],
        "outstanding_loans": input_data['Outstanding Loans'],
        "financial_insights": input_data['Financial Insights'],
        "ai_think": ai_think
    }
    
    return loan_analysis_data

# Insert data into Supabase
def insert_loan_analysis_data(loan_analysis_data):

    # Insert into 'loan_analysis' table
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

# Function to generate financial insights
def generate_financial_insights(merged_data):
    prompt = FINANCIAL_INSIGHTS_PROMPT

    chat = ChatGroq(model_name=FINANCIAL_INSIGHTS_MODEL)
    messages = [HumanMessage(content=prompt)]  # Use HumanMessage here instead of SystemMessage
    
    response = chat(messages).content.strip()
   
    think = extract_think_content(response)
    # Extract financial insights and loan eligibility separately
    insights, eligibility = extract_insights_le(response)
    merged_data["Financial Insights"] = insights
    merged_data["Loan Eligibility"] = eligibility
    return merged_data, think



def extract_insights_le(insights):
       # Regular expression to capture everything after "### Financial Insight" and before "### Loan Eligibility"
    pattern = re.compile(r"### Financial Insights:\s*(.*?)\s*### Loan Eligibility:\s*(true|false|yes|no)", re.DOTALL | re.IGNORECASE)

    # Search for the pattern in the insights response
    match = pattern.search(insights)

    if match:
        financial_insights = match.group(1).strip()  # Extract financial insight text
        loan_eligibility = match.group(2).strip().lower()  # Normalize eligibility result
        return financial_insights, loan_eligibility
    else:
        return "Unknown", "Unknown"

def extract_think_content(input_text):
    # Regex to extract <think> content
    think_pattern = r'<think>\s*([\s\S]+?)\s*</think>'
    
    # Search for the <think> content in the input text
    think_match = re.search(think_pattern, input_text)
    return think_match.group(1) if think_match else None
    
def extract_json_from_input(input_text):
    # Regex to extract JSON part after <think> tags
    json_pattern = r'```json\n([\s\S]+?)\n```'

    # Search for the JSON content in the input text
    match = re.search(json_pattern, input_text)
    
    if match:
        json_str = match.group(1)  # Extract JSON string
        try:
            # Parse the extracted JSON string into a Python dictionary
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError:
            return ""
    else:
        return ""

def retrieve_document_by_type_and_id(doc_type, file_name):
    print("Retrieving relevant document...")
    print(doc_type)
    print(file_name)
    response = (
        supabase_client.table('file_embeddings')
        .select('text')
        .eq('type', doc_type)
        .eq('file_name', file_name)
        .execute()
    )

    result = response.data[0]['text'].strip() if response.data else ""
    length = len(result)
    diff = length - 20000 
    return result[diff:] if len(result) > 20000 else result


def retrieve_relevant_document(qe):
    # Ensure the query embedding has the right shape
    query_embedding = np.array(qe).reshape(1, -1)
    
    response = supabase_client.table('file_embeddings').select('file_id', 'file_name', 'text', 'embedding').execute()
    
    best_score = -1
    best_context = None
    
    # Iterate over all documents in Supabase
    print("Retrieving relevant document...")
    for row in response.data:
        # Convert the string representation of the list into an actual list and reshape
        stored_vector = np.array(ast.literal_eval(row['embedding'])).reshape(1, -1)
        similarity_score = cosine_similarity(query_embedding, stored_vector)[0][0]
        
        if similarity_score > best_score:
            best_score = similarity_score
            best_context = row['text'].strip()  # Most relevant document content
    
    return best_context


def fetch_previous_answer(loan_id, file_type):
    # Query the loan_chat_history table to check if the answer already exists
    response = supabase_client.table("loan_chat_history").select("answer").eq("loan_id", loan_id).eq("file_type", file_type).execute()

    if response.data:
        # If a previous answer is found, return it
        return response.data[0]["answer"]
    else:
        return None

def store_answer_in_history(loan_id, file_type, query, answer):
    # Store the new answer in loan_chat_history
    response = supabase_client.table("loan_chat_history").insert({
        "loan_id": loan_id,
        "file_type": file_type,
        "query": query,
        "answer": answer
    }).execute()

# Use LangChain to combine document retrieval and answer generation
def generate_answer(file_type, file_name, loan_id):
     query = JSON_EXTRACTOR_PROMPT
     print("hello")
     print(query)
     previous_answer = fetch_previous_answer(loan_id,file_type)
     if previous_answer:
        print("Returning cached answer...")
        return previous_answer
     else:
        llm = ChatGroq(model=FINANCIAL_INSIGHTS_MODEL)  # Initialize ChatGroq with your API key
                    # Define the prompt for ChatGroq (RAG-based)
        prompt_template = """Based on the following, answer the user's question:
            Document Content:
            {context}
            User Question: {query}
            Answer:
        """
        prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=prompt_template
        )
        relevant_context = retrieve_document_by_type_and_id(file_type, file_name)
        prompt_input = prompt.format(context=relevant_context, query=query)
        response = llm.invoke([{"role": "system", "content": prompt_input}])
        answer = response.content
        store_answer_in_history(loan_id, file_type, query, answer)
        return answer

def merge_details(*details_list):
    merged = {
        "Total Monthly Deposits": 0,
        "Total Monthly Withdrawals": 0,
        "Regular Bills": [],
        "Outstanding Loans": []
    }

    bill_tracker = defaultdict(float)  # To track summed bill amounts

    for details in details_list:
        merged["Total Monthly Deposits"] += details["Total Monthly Deposits"]
        merged["Total Monthly Withdrawals"] += details["Total Monthly Withdrawals"]

        for bill in details["Regular Bills"]:
            bill_tracker[bill["type"]] += bill["amount"]

        merged["Outstanding Loans"].extend(details["Outstanding Loans"])

    # Convert merged bills back into list format
    merged["Regular Bills"] = [{"type": t, "amount": amt, "frequency": "Monthly"} for t, amt in bill_tracker.items()]

    return merged

def do_loan_analysis(loan_id):
    print(f"Starting loan analysis for Loan ID: {loan_id}")
    # Generate answers from different document parts
    json_from_txt = extract_json_from_input(generate_answer("txt", f"{loan_id}.pdf", loan_id))
    print(json_from_txt)
    json_from_img = extract_json_from_input(generate_answer("img_txt", f"{loan_id}.pdf", loan_id))
    json_from_table = extract_json_from_input(generate_answer("table_txt", f"{loan_id}.pdf", loan_id))

    print("Merging JSON objects...")
    merged_data = merge_details(json_from_txt, json_from_img, json_from_table)

    print(merged_data)

    print("Generating financial insights...")
    merged_data, think = generate_financial_insights(merged_data)

    print("Converting to loan analysis format...")
    loan_analysis_data = convert_to_loan_analysis_data(loan_id, merged_data, think)

    print("Inserting loan analysis data into the database...")
    insert_loan_analysis_data(loan_analysis_data)

    print("Loan analysis completed successfully.")