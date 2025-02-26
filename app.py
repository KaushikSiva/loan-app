from flask import Flask, request, jsonify, render_template
from supabase import create_client, Client
import os
import pdf_embedder
import retreiver
import asyncio
import uuid
from flask_cors import CORS  # Import CORS


app = Flask(__name__)

# Initialize Supabase client with your credentials (ensure to set your URL and API key)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('index2.html')

@app.route('/loan-analysis/<int:loan_id>')
def get_loan_analysis(loan_id):
    # Fetch loan analysis data from Supabase
    analysis_response = supabase.table('loan_analysis').select('*').eq('loan_id', loan_id).execute()
    

    if analysis_response.data:
        analysis = analysis_response.data[0]
        print(analysis["ai_decision"])
        # Fetch loan details (business name, principal owner name, and date) from the loan table
        loan_response = supabase.table('loan').select('business_name', 'principal_owner_name', 'date').eq('loan_id', loan_id).execute()

        if loan_response.data:
            loan = loan_response.data[0]
            
            # Combine loan analysis with loan details
            return jsonify({
                'loan_analysis_id': analysis['loan_analysis_id'],
                'loan_id': analysis['loan_id'],
                'manual_loan_approval': analysis['manual_loan_approval'],
                'manual_decision_reason': analysis['manual_decision_reason'],
                'ai_decision': analysis['ai_decision'],
                'ai_think': analysis['ai_think'],
                'total_monthly_deposits': analysis['total_monthly_deposits'],
                'total_monthly_withdrawals': analysis['total_monthly_withdrawals'],
                'regular_bills': analysis['regular_bills'],
                'outstanding_loans': analysis['outstanding_loans'],
                'financial_insights': analysis['financial_insights'],
                'business_name': loan['business_name'],
                'principal_owner_name': loan['principal_owner_name'],
                'date': loan['date']
            })
        return jsonify({'message': 'Loan details not found'}), 404
    
    return jsonify({'message': 'Loan analysis not found'}), 404

@app.route('/approve-loan', methods=['POST'])
def approve_loan():
    data = request.json
    print("hi")
    print(data)
    print("h2")
    manual_loan_approval = data['manual_loan_approval']
    manual_decision_reason = data['manual_decision_reason']

    loan_id = data['loan_id']
    # Update the loan analysis data in Supabase
    response = supabase.table('loan_analysis').update({
        'manual_loan_approval': manual_loan_approval,
        "manual_decision_reason" : manual_decision_reason
    }).eq('loan_id', loan_id).execute()
    if response.data:
        return jsonify({'message': 'Loan analysis updated successfully'}), 200
    return jsonify({'message': 'Failed to update loan analysis'}), 500


@app.route('/upload-file/<loan_id>', methods=['POST'])
def upload_file(loan_id):
    if not loan_id.isdigit():
        return jsonify({"error": "Invalid loan ID"}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filename = f"{loan_id}{os.path.splitext(file.filename)[-1]}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)  # Save file immediately

    # Run async processing in the background
    asyncio.run(process_file_async(file_path, loan_id))
    return jsonify({"message": "File uploaded successfully. Processing in background."}), 200

async def process_file_async(file_path, loan_id):
    # Run your file processing logic here
    uuid = generate_uuid_from_file_name(file_path)
    pdf_embedder.process_file(uuid, loan_id)

def generate_uuid_from_file_name(input_string):
    namespace = uuid.NAMESPACE_DNS  # You can use NAMESPACE_DNS, NAMESPACE_URL, or a custom UUID
    return str(uuid.uuid5(namespace, input_string))

@app.route('/loan-analysis/<loan_id>', methods=['POST'])
def loan_analysis(loan_id):
    retreiver.do_loan_analysis(loan_id)
    return jsonify({"message": "Triggered Loan analysis.It will be done in the background"}), 200

if __name__ == '__main__':  # Default to 5000, but Render assigns dynamically
    app.run(debug=True)