from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend_college():
    data = request.get_json()

    # Normalize inputs
    fees_map = {"low": 40000, "medium": 70000, "high": 100000}
    rating_ranges = {
        "low": (0, 3.0),
        "medium": (3.0, 4.5),
        "high": (4.5, 5.0)
    }

    fees_value = fees_map.get(data["fees"].lower(), 70000)
    placement_range = rating_ranges.get(data["placement_rating"].lower(), (3.0, 4.5))
    overall_range = rating_ranges.get(data["overall_rating"].lower(), (3.0, 4.5))
    hostel = 1 if data["hostel"] else 0

    user_type = data["type"].strip().lower()
    user_location = data["location"].strip().lower()

    # Load and preprocess the dataset
    df = pd.read_csv("cleaned_college_data.csv")
    df['Type'] = df['Type'].str.strip().str.lower()
    df['Location'] = df['Location'].str.strip().str.lower()

    # Filter by location and rating ranges
    df_filtered = df[
        (df['Location'] == user_location) &
        (df['Placement Rating'] >= placement_range[0]) &
        (df['Placement Rating'] <= placement_range[1]) &
        (df['Overall Rating'] >= overall_range[0]) &
        (df['Overall Rating'] <= overall_range[1])
    ]

    if df_filtered.empty:
        return jsonify({"error": "No colleges found matching your criteria"}), 404

    # Encode type
    le_type = LabelEncoder()
    df_filtered['Type_encoded'] = le_type.fit_transform(df_filtered['Type'])
    user_type_encoded = le_type.transform([user_type])[0]

    # Create feature matrix
    X = df_filtered[["Fees", "Hostel", "Placement Rating", "Overall Rating", "Type_encoded"]]
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(X)

    user_vector = [[fees_value, hostel, 
                    sum(placement_range)/2,  # Use mid-point of range for comparison
                    sum(overall_range)/2,
                    user_type_encoded]]

    distances, indices = knn.kneighbors(user_vector)

    recommended_index = indices[0][0]
    recommended_college = df_filtered.iloc[recommended_index]["College Name"]

    return jsonify({"recommendations": [recommended_college]})

if __name__ == "__main__":
    app.run(debug=True)