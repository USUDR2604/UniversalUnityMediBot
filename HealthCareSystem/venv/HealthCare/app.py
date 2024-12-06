from flask import Flask, request, jsonify, render_template
from response_generator import generate_response  # Import the modified function
import nltk
import numpy as np

nltk.data.path.append('/Users/udaydeepreddy/Desktop/HealthCareSystem/venv/HealthCare/nltk_data')

app = Flask(__name__)

# Helper function to convert non-serializable data to JSON-compatible types
def make_json_serializable(data):
    if isinstance(data, dict):
        return {k: make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_serializable(item) for item in data]
    elif isinstance(data, np.int64):  # Convert np.int64 to int
        return int(data)
    elif isinstance(data, np.ndarray):  # Convert numpy arrays to lists
        return data.tolist()
    else:
        return data

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_input = request.get_json().get("message")
    
    # Call generate_response and capture the returned values
    response_message, entity_analysis, context_analysis, tag_info, encoder_info, vector_info = generate_response(user_input)
    
    # Prepare the data dictionary
    data = {
        "response": response_message,
        "entity_analysis": entity_analysis,
        "context_analysis": context_analysis,
        "tag_info": tag_info,
        "encoder_info": encoder_info,
        "vector_info": vector_info
    }
    
    # Ensure all data is JSON serializable
    serializable_data = make_json_serializable(data)
    
    # Return each value in the JSON response
    return jsonify(serializable_data)

if __name__ == "__main__":
    app.run(debug=True)
