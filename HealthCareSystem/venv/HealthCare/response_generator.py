import random
import pickle
import json
import re
from pymongo import MongoClient
import warnings
warnings.filterwarnings("ignore")

# Load pre-trained model components
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Reset function to clear session-specific data


def reset_session():
    global last_patient_id
    last_patient_id = None  # Clear the stored patient ID


# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["HealthCareDataCenter"]
patients_collection = db["PatientDetails"]

# Load intents and context mappings
with open("intents.json") as file:
    intents = json.load(file)
responses = {intent['tag']: intent['responses']
             for intent in intents['intents']}
context_mapping = {intent['tag']: intent['context'][0]
                   if intent['context'] else "" for intent in intents['intents']}

# Store the last recognized Patient ID for context
last_patient_id = None

# Function to extract a 12-character alphanumeric Patient ID


def extract_patient_id(user_input):
    match = re.search(r'\b[A-Za-z0-9]{12}\b', user_input)
    return match.group(0) if match else None

# Main function to perform action based on context


def perform_action(context, patient_id=None):
    context_actions = {
        "fetch_medical_condition": fetch_medical_condition,
        "fetch_medication": fetch_medication,
        "fetch_billing_info": fetch_billing_info,
        "fetch_test_results": fetch_test_results,
        "fetch_insurance_info": fetch_insurance_info,
        "fetch_admission_info": fetch_admission_info,
        "fetch_discharge_info": fetch_discharge_info,
        "fetch_patient_details": fetch_patient_details
    }
    action_function = context_actions.get(context)
    return action_function(patient_id) if action_function else "No action found for the current context."


# Define functions to fetch patient details and related information from MongoDB
def fetch_patient_details(patient_id):
    result = patients_collection.find_one(
        {"Patient ID": patient_id}, {"_id": 0})
    if result:
        details = (
            f"Patient Details for ID {patient_id}:\n"
            f"Name: {result.get('Name', 'N/A')}\n"
            f"Age: {result.get('Age', 'N/A')}\n"
            f"Gender: {result.get('Gender', 'N/A')}\n"
            f"Blood Type: {result.get('Blood Type', 'N/A')}\n"
            f"Medical Condition: {result.get('Medical Condition', 'N/A')}\n"
            f"Date of Admission: {result.get('Date of Admission', 'N/A')}\n"
            f"Doctor: {result.get('Doctor', 'N/A')}\n"
            f"Hospital: {result.get('Hospital', 'N/A')}\n"
            f"Insurance Provider: {result.get('Insurance Provider', 'N/A')}\n"
            f"Billing Amount: ${result.get('Billing Amount', 'N/A'):.2f}\n"
            f"Room Number: {result.get('Room Number', 'N/A')}\n"
            f"Admission Type: {result.get('Admission Type', 'N/A')}\n"
            f"Discharge Date: {result.get('Discharge Date', 'N/A')}\n"
            f"Medication: {result.get('Medication', 'N/A')}\n"
            f"Test Results: {result.get('Test Results', 'N/A')}"
        )
        return f"Here are the details for Patient ID {patient_id} 😊\n\n{details}"
    else:
        return f"No records found for Patient ID {patient_id}. ❗️"


def fetch_medical_condition(patient_id):
    result = patients_collection.find_one({"Patient ID": patient_id}, {
                                          "Medical Condition": 1, "_id": 0})
    return f"Medical Condition for Patient ID {patient_id}: {result['Medical Condition']} ✅" if result else f"No medical condition record found for Patient ID {patient_id}. ❗️"


def fetch_medication(patient_id):
    result = patients_collection.find_one(
        {"Patient ID": patient_id}, {"Medication": 1, "_id": 0})
    return f"Medication for Patient ID {patient_id}: {result['Medication']} 💊" if result else f"No medication records found for Patient ID {patient_id}. ❗️"


def fetch_billing_info(patient_id):
    result = patients_collection.find_one({"Patient ID": patient_id}, {
                                          "Billing Amount": 1, "_id": 0})
    return f"Billing Amount for Patient ID {patient_id}: ${result['Billing Amount']:.2f} 💵" if result else f"No billing information found for Patient ID {patient_id}. ❗️"


def fetch_test_results(patient_id):
    result = patients_collection.find_one({"Patient ID": patient_id}, {
                                          "Test Results": 1, "_id": 0})
    return f"Test Results for Patient ID {patient_id}: {result['Test Results']} 🧪" if result else f"No test results found for Patient ID {patient_id}. ❗️"


def fetch_insurance_info(patient_id):
    result = patients_collection.find_one({"Patient ID": patient_id}, {
                                          "Insurance Provider": 1, "_id": 0})
    return f"Insurance Provider for Patient ID {patient_id}: {result['Insurance Provider']} 🏥" if result else f"No insurance information found for Patient ID {patient_id}. ❗️"


def fetch_admission_info(patient_id):
    result = patients_collection.find_one({"Patient ID": patient_id}, {
                                          "Date of Admission": 1, "Admission Type": 1, "_id": 0})
    if result:
        return (f"Admission Details for Patient ID {patient_id}:\n"
                f"Date of Admission: {result['Date of Admission']}\n"
                f"Admission Type: {result['Admission Type']} 🏥")
    else:
        return f"No admission information found for Patient ID {patient_id}. ❗️"


def fetch_discharge_info(patient_id):
    result = patients_collection.find_one({"Patient ID": patient_id}, {
                                          "Discharge Date": 1, "_id": 0})
    return f"Discharge Date for Patient ID {patient_id}: {result['Discharge Date']} 🏥" if result else f"No discharge information found for Patient ID {patient_id}. ❗️"


def fetch_blood_pressure(patient_id):
    result = patients_collection.find_one({"Patient ID": patient_id}, {
                                          "Blood Pressure": 1, "_id": 0})
    return f"Blood Pressure result for Patient ID {patient_id}: {result['Blood Pressure']} 💉" if result else f"No blood pressure records found for Patient ID {patient_id}. ❗️"


# Keyword groups for actions
keyword_to_function = {
    "patient_details": (["patient", "patient details", "full details", "profile", "info", "information"], fetch_patient_details),
    "medical_condition": (["medical condition", "diagnosis", "condition", "disease"], fetch_medical_condition),
    "medication": (["medication", "drugs", "prescription", "medicine", "meds"], fetch_medication),
    "billing_amount": (["billing amount", "bill", "cost", "charges", "total bill"], fetch_billing_info),
    "test_results": (["test results", "lab results", "tests", "diagnostic", "results"], fetch_test_results),
    "insurance_provider": (["insurance provider", "insurer", "insurance", "coverage", "insurance company"], fetch_insurance_info),
    "admission_info": (["admission info", "admission date", "admission details", "admission"], fetch_admission_info),
    "discharge_date": (["discharge date", "discharge", "date of discharge", "release date"], fetch_discharge_info),
    "blood_pressure": (["blood pressure", "bp", "pressure", "blood pressure results", "bp data"], fetch_blood_pressure)
}

# Define patient-related keywords based on categories
patient_related_keywords = (
    keyword_to_function["patient_details"][0]
    + keyword_to_function["medical_condition"][0]
    + keyword_to_function["billing_amount"][0]
    + keyword_to_function["test_results"][0]
)

non_patient_modules = {
    "pharmacy": {
        "keywords": ["pharmacy", "find pharmacy", "pharmacy nearby"],
        "response": "Please provide the pharmacy name or location to assist you with nearby options.",
        "context": "Pharmacy search initiated"
    },
    "hospital": {
        "keywords": ["hospital", "find hospital", "hospital nearby"],
        "response": "Please provide the hospital name or type (e.g., general, specialty) and location.",
        "context": "Hospital search initiated"
    },
    "adverse_drug": {
        "keywords": ["adverse drug", "adverse drugs", "drug effects", "adverse reactions"],
        "response": "Navigating to Adverse Drug Reactions module. Please specify if you'd like to see details for a specific drug or patient.",
        "context": "Adverse drug module initiated"
    }
}


def generate_response(user_input):
    global last_patient_id
    entity_analysis = "None"  # Default value for entity analysis
    context_analysis = "None"  # Default value for context analysis

    # Preprocess user input and vectorize it
    processed_input = user_input.lower()
    input_vector = vectorizer.transform([processed_input])

    # Predict the tag for the input
    predicted_tag = best_model.predict(input_vector)[0]
    tag_info = f"Predicted Tag: {predicted_tag}"

    # Decode predicted tag to human-readable label
    # Decodes to human-readable label if encoded
    encoder_info = encoder.inverse_transform([predicted_tag])[0]
    vector_info = input_vector.toarray()  # Represents vectorized user input

    # Handle greeting specifically
    if encoder_info == "greeting":
        response_text = random.choice(
            ["Hello! How can I assist you today? 😊", "Hi there! What can I help you with today?"])
        context_analysis = "Greeting handled"
        return response_text, entity_analysis, context_analysis, tag_info, encoder_info, vector_info

    # Check if user wants to reset session
    if user_input.lower() == "clear":
        reset_session()  # Clear stored data
        response_text = "Session has been reset. You can start again."
        return response_text, entity_analysis, "Session cleared", tag_info, encoder_info, vector_info

    # Non-Patient-ID Queries
    for module, details in non_patient_modules.items():
        if any(keyword in user_input.lower() for keyword in details["keywords"]):
            response_text = details["response"]
            context_analysis = details["context"]
            return response_text, entity_analysis, context_analysis, tag_info, encoder_info, vector_info

    # Extract Patient ID if provided
    patient_id = extract_patient_id(user_input)
    if patient_id:
        last_patient_id = patient_id
        response_text = f"Patient ID {patient_id} recognized. What details would you like to retrieve? Options include: Medication, Condition, Billing, etc."
        entity_analysis = f"Recognized Patient ID: {patient_id}"
        context_analysis = "Patient ID recognized and stored"
        return response_text, entity_analysis, context_analysis, tag_info, encoder_info, vector_info

    # Prompt for Patient ID if patient-specific details are requested without an ID
    if last_patient_id is None and any(keyword in processed_input for keyword in patient_related_keywords):
        response_text = "Please provide the Patient ID to proceed with patient-specific information."
        context_analysis = "Awaiting Patient ID for patient-specific request"
        return response_text, entity_analysis, context_analysis, tag_info, encoder_info, vector_info

    # Use last recognized Patient ID for patient-specific requests
    if last_patient_id:
        for detail_name, (keywords, fetch_function) in keyword_to_function.items():
            if any(keyword in processed_input for keyword in keywords):
                result = fetch_function(last_patient_id)
                context_analysis = f"Fetched {detail_name} for Patient ID {last_patient_id}"
                return result, entity_analysis, context_analysis, tag_info, encoder_info, vector_info

    # Default fallback response
    response_text = "I'm here to assist with healthcare-related queries. Please specify how I can help."
    context_analysis = "No specific request matched, prompted for more info"

    return response_text, entity_analysis, context_analysis, tag_info, encoder_info, vector_info
