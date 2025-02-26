from flask import Flask, request, jsonify, render_template
from supabase import create_client, Client
import os
import asyncio
import uuid


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

if __name__ == '__main__':  # Default to 5000, but Render assigns dynamically
     port = int(os.environ.get("PORT", 5000))  # Default to 5000, but Render assigns dynamically
     app.run(host='0.0.0.0', port=port, debug=True)