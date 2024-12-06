import pandas as pd
from pymongo import MongoClient

# Load the CSV file into a DataFrame
data = pd.read_csv('hospital_locations.csv')
# Convert the DataFrame to a list of dictionaries for MongoDB insertion
data_dict = data.to_dict(orient="records")
# Connect to MongoDB
try:
    client = MongoClient("mongodb://localhost:27017/")  
    # Replace with your MongoDB URI if using a remote server
    db = client["HealthCareDataCenter"]  # Database name
    collection = db["HospitalLocationDetails"]  # Collection name
    # Insert all data into MongoDB
    result = collection.insert_many(data_dict)
    # Confirmation message
    print(f"Inserted {len(result.inserted_ids)} records into the 'PatientDetails' collection")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    client.close()
