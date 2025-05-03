# -*- coding: utf-8 -*-
"""
Created on Fri May  2 14:12:01 2025

@author: vargh
"""

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load and preprocess the data
df = pd.read_csv("cleaned_college_data.csv")

# Encode 'Type' and 'Location'
le_type = LabelEncoder()
le_location = LabelEncoder()

df['Type_encoded'] = le_type.fit_transform(df['Type'])
df['Location_encoded'] = le_location.fit_transform(df['Location'])

# Final feature set
features = ['Fees(aprox)', 'Hostel', 'Placement', 'Overall', 'Type_encoded', 'Location_encoded']
X = df[features]
college_names = df['College']

# Fit the KNN model
knn = NearestNeighbors(n_neighbors=3)
knn.fit(X)

@app.route('/')
def home():
    return "College Recommendation API is running!"

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    preferences = data.get('preferences', [])

    if not preferences or len(preferences) != len(features):
        return jsonify({"error": f"Send exactly {len(features)} preference values."}), 400

    # Find nearest colleges
    distances, indices = knn.kneighbors([preferences])
    recommendations = college_names.iloc[indices[0]].tolist()

    return jsonify({
        "preferences_received": preferences,
        "recommendations": recommendations
    })

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)





