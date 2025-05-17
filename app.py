from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and preprocess the data
df = pd.read_csv("cleaned_college_data.csv")

# Encode categorical features
le_type = LabelEncoder()
le_location = LabelEncoder()
df['Type_encoded'] = le_type.fit_transform(df['Type'])
df['Location_encoded'] = le_location.fit_transform(df['Location'])

# Feature set for recommendations
features = ['Fees(aprox)', 'Hostel', 'Placement', 'Overall', 'Type_encoded', 'Location_encoded']
X = df[features]
college_names = df['College']

# Train the KNN model
knn = NearestNeighbors(n_neighbors=3, algorithm='auto')
knn.fit(X)

# Conversion functions
def convert_fees(fee):
    try:
        fee = float(fee)
        if fee <= 40000:
            return 40000
        elif fee <= 80000:
            return 80000
        return 150000
    except ValueError:
        return 40000 if fee.lower() == "low" else (80000 if fee.lower() == "medium" else 150000)

def convert_rating(rating, thresholds=(2.5, 4.0, 4.8)):
    try:
        rating = float(rating)
        if rating < thresholds[0]:
            return thresholds[0]  # low
        elif rating < thresholds[1]:
            return thresholds[1]  # medium
        return thresholds[2]  # high
    except ValueError:
        return thresholds[0] if rating.lower() == "low" else (thresholds[1] if rating.lower() == "medium" else thresholds[2])

@app.route('/')
def home():
    return jsonify({"message": "College Recommendation API is running!"})

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()

    # Extract and convert user inputs
    fees = convert_fees(data.get('fees'))
    hostel = 1 if data.get('hostel') else 0
    placement = convert_rating(data.get('placement_rating'))
    overall = convert_rating(data.get('overall_rating'))
    college_type = data.get('type')
    location = data.get('location')

    # Validate inputs
    if college_type not in le_type.classes_ or location not in le_location.classes_:
        return jsonify({"error": "Invalid type or location"}), 400

    # Encode categorical inputs
    type_encoded = le_type.transform([college_type])[0]
    location_encoded = le_location.transform([location])[0]

    preferences = [fees, hostel, placement, overall, type_encoded, location_encoded]

    # Get recommendations
    distances, indices = knn.kneighbors([preferences])
    recommendations = college_names.iloc[indices[0]].tolist()

    return jsonify({"recommendations": recommendations})

import os

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 10000)), debug=False, use_reloader=False)