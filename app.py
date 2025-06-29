from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend_college():
    data = request.get_json()

    # Maps
    fees_map = {"low": 40000, "medium": 70000, "high": 100000}
    rating_ranges = {
        "low": (0, 3.0),
        "medium": (3.0, 4.5),
        "high": (4.5, 5.0)
    }
    type_map = {
        "private": 0,
        "government": 1
    }

    # Convert input
    fees_value = fees_map.get(data["fees"].lower(), 70000)
    placement_range = rating_ranges.get(data["placement_rating"].lower(), (3.0, 4.5))
    overall_range = rating_ranges.get(data["overall_rating"].lower(), (3.0, 4.5))
    hostel = 1 if data["hostel"] else 0
    user_type = data["type"].strip().lower()
    user_location = data["location"].strip().lower()

    user_type_encoded = type_map.get(user_type)
    if user_type_encoded is None:
        return jsonify({"error": "Invalid college type"}), 400

    # Load dataset
    df = pd.read_csv("cleaned_college_data.csv")

    # Preprocess
    df['Location'] = df['Location'].str.strip().str.lower()
    df['Type'] = df['Type'].str.strip().str.lower()
    df['Type_encoded'] = df['Type'].map(type_map)

    # Filter by location and rating ranges
    df_filtered = df[
        (df['Location'] == user_location) &
        (df['Placement'] >= placement_range[0]) &
        (df['Placement'] <= placement_range[1]) &
        (df['Overall'] >= overall_range[0]) &
        (df['Overall'] <= overall_range[1])
    ]

    if df_filtered.empty:
        return jsonify({"error": "No colleges found matching your criteria"}), 404

    # Build KNN
    X = df_filtered[["Fees(aprox)", "Hostel", "Placement", "Overall", "Type_encoded"]]
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(X)

    # Create user vector using midpoints of rating ranges
    user_vector = [[
        fees_value,
        hostel,
        sum(placement_range) / 2,
        sum(overall_range) / 2,
        user_type_encoded
    ]]

    distances, indices = knn.kneighbors(user_vector)
    recommended_index = indices[0][0]
    recommended_college = df_filtered.iloc[recommended_index]["College"]

    return jsonify({"recommendations": [recommended_college]})

if __name__ == "__main__":
    app.run(debug=True)