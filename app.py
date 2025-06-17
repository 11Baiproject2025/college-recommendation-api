from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load and preprocess the data
df = pd.read_csv("cleaned_college_data.csv")

# Encode categorical features
le_type = LabelEncoder()
le_location = LabelEncoder()

df['Type_cleaned'] = df['Type'].str.lower().str.strip()
df['Location_cleaned'] = df['Location'].str.lower().str.strip()

le_type.fit(df['Type_cleaned'])
le_location.fit(df['Location_cleaned'])

df['Type_encoded'] = le_type.transform(df['Type_cleaned'])
df['Location_encoded'] = le_location.transform(df['Location_cleaned'])

# Feature set
features = ['Fees(aprox)', 'Hostel', 'Placement', 'Overall', 'Type_encoded', 'Location_encoded']
X = df[features]
college_names = df['College']

# Train the KNN model
knn = NearestNeighbors(n_neighbors=3, algorithm='auto')
knn.fit(X)

# Convert inputs
def convert_fees(fee):
    try:
        fee = float(fee)
        if fee <= 40000:
            return 40000
        elif fee <= 80000:
            return 80000
        return 150000
    except ValueError:
        fee = fee.lower()
        return 40000 if fee == "low" else 80000 if fee == "medium" else 150000

def convert_rating(rating, thresholds=(2.5, 4.0, 4.8)):
    try:
        rating = float(rating)
        if rating < thresholds[0]:
            return thresholds[0]
        elif rating < thresholds[1]:
            return thresholds[1]
        return thresholds[2]
    except ValueError:
        rating = rating.lower()
        return thresholds[0] if rating == "low" else thresholds[1] if rating == "medium" else thresholds[2]

@app.route('/')
def home():
    return jsonify({"message": "College Recommendation API is running!"})

@app.route('/recommend', methods=['POST'])

def recommend():

    data = request.get_json()



    # Extract and convert inputs

    fees = convert_fees(data.get('fees'))

    hostel = 1 if data.get('hostel') == 'true' else 0

    placement = convert_rating(data.get('placement'))

    overall = convert_rating(data.get('overall'))

    college_type = data.get('type', '').lower().strip()

    location = data.get('location', '').lower().strip()



    # Validate type and location

    if college_type not in le_type.classes_ or location not in le_location.classes_:

        return jsonify({

            "error": "Invalid type or location",

            "valid_types": le_type.classes_.tolist(),

            "valid_locations": le_location.classes_.tolist()

        }), 400



    # (Your model prediction logic here — example below)

    # Just sending a dummy college for now:

    return jsonify({

        "college": "Model College of Engineering"

    })

    # Encode type and location
    type_encoded = le_type.transform([college_type])[0]
    location_encoded = le_location.transform([location])[0]

    preferences = [fees, hostel, placement, overall, type_encoded, location_encoded]

    # Get recommendations
    distances, indices = knn.kneighbors([preferences])
    recommendations = college_names.iloc[indices[0]].tolist()

    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 10000)), debug=False, use_reloader=False)