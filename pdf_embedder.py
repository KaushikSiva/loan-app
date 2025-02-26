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
import uuid
from dotenv import load_dotenv



load_dotenv()

GROK_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
HF_TOKEN = os.getenv("HF_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
IMAGE_TO_TEXT_MODEL = os.getenv("IMAGE_TO_TEXT_MODEL")
TABLE_TO_TEXT_MODEL = os.getenv("TABLE_TO_TEXT_MODEL")
TABLE_TO_TEXT_PROMPT = os.getenv("TABLE_TO_TEXT_PROMPT")
IMAGE_TO_TEXT_PROMPT = os.getenv("IMAGE_TRANSACTION_SUMMARY_PROMPT")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def partition_input(fp):

# Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
    chunks = partition_pdf(
        filename=fp,
        infer_table_structure=True,            # extract tables
        strategy="hi_res",                     # mandatory to infer tables

        extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
        # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

        extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

        chunking_strategy="by_title",          # or 'basic'
        max_characters=10000,                  # defaults to 500
        combine_text_under_n_chars=2000,       # defaults to 0
        new_after_n_chars=6000,

        # extract_images_in_pdf=True,          # deprecated
    )
    return chunks

def separate_tables_from_texts(parts):
    # separate tables from texts
    tables = []
    texts = []
    for chunk in parts:
        if "Table" in str(type(chunk)):
            tables.append(chunk)

        if "CompositeElement" in str(type((chunk))):
            texts.append(chunk) 
    return tables, texts

# Get the images from the CompositeElement objects
def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

def convert_text_table_to_text(texts, tables):
    # Prompt
    prompt_text = TABLE_TO_TEXT_PROMPT
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatGroq(temperature=0.5, model=TABLE_TO_TEXT_MODEL)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    # Summarize text
    txt = summarize_chain.batch(texts, {"max_concurrency": 1})

    concatenated_text = "".join(txt)


    # Summarize tables
    tables_html = [table.metadata.text_as_html for table in tables]
    table_text = summarize_chain.batch(tables_html, {"max_concurrency": 1})
    concatenated_table_text = "".join(table_text)
    return concatenated_text, concatenated_table_text

def convert_image_to_text(base64_image):
    # Define the prompt template
    prompt_template = IMAGE_TO_TEXT_PROMPT

    # Initialize the ChatGroq model
    chat = ChatGroq(model=IMAGE_TO_TEXT_MODEL)

    # Prepare the message with the base64 encoded image
    message = HumanMessage(
        content=[
            {"type": "text", "text":prompt_template},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    )

    # Get the response from the model
    response = chat.invoke([message])
    image_text = response.content
    return image_text

def get_embedding(text):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embedding = model.encode(text)
    return embedding.tolist()


def store_document_embedding(file_id, file_name, file_type, file_txt, loan_id):
    document_text = file_txt.strip()
    embedding = get_embedding(document_text)
    
    # Store the document embedding and text in Supabase
    supabase_client.table('file_embeddings').upsert({
        "file_id": file_id,
        "file_name": file_name,
        "embedding": embedding,
        "type": file_type,
        "text": document_text,
        "loan_id": loan_id
    }).execute()


def run_workflow(uuid, loan_id):
    file_name = f"{loan_id}.pdf"
    seperator = "\n\n\n"
    output_path = "/tmp/uploads/"
    file_path = output_path + file_name
    print("Partitioning the file")
    chunks = partition_input(file_path)
    print("Separating tables from texts")
    tables, texts = separate_tables_from_texts(chunks)
    print("Converting tables to text")
    txt, table_txt = convert_text_table_to_text(texts, tables)
    images_text = ""
    images = get_images_base64(chunks)
    img_len = len(images)
    cnt = 0
    print("Converting images to text")
    for image in images:
        cnt += 1
        print(f"Processing image {cnt} of {img_len}")
        images_text += convert_image_to_text(image)

    store_document_embedding(uuid, file_name, "txt", txt, loan_id)
    store_document_embedding(uuid, file_name, "img_txt", images_text, loan_id)
    store_document_embedding(uuid, file_name, "table_txt", table_txt, loan_id)


def process_file(uuid, loan_id):
    print("Processing file")
    run_workflow(uuid, loan_id)
    print("Embedding Done")

