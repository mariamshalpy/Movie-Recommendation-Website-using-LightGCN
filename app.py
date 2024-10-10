from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import torch
from model import (
    load_data,
    LightGCN,
    preprocess_data,
    get_top_recommendations,
    update_model_with_new_rating,
)

app = Flask(__name__)
app.secret_key = "your_secret_key"

global model

# Load model and data
movies_df, ratings_df = load_data()
train_df, test_df, n_users, n_items, le_user, le_item = preprocess_data(ratings_df)
model = LightGCN(n_users, n_items, latent_dim=64, num_layers=3).to("cpu")
model.load_state_dict(torch.load("lightgcn_model.pth", map_location="cpu"))
model.eval()


@app.route("/")
def index():
    available_genres = list(set("|".join(movies_df["genres"]).split("|")))
    return render_template("index.html", genres=available_genres)


@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = request.form.get("user_id")
    selected_genres = request.form.get("genres").split(",")

    recommendations, _ = get_top_recommendations(
        int(user_id),
        selected_genres,
        model,
        movies_df,
        ratings_df,
        le_user,
        le_item,
        n_users,
    )

    print(recommendations.to_dict(orient="records"))  # Add this line

    return render_template(
        "results.html", recommendations=recommendations.to_dict(orient="records")
    )


@app.route("/rate", methods=["POST"])
def rate():
    global model, ratings_df, n_users  # Declare model, ratings_df, and n_users as global
    user_id = request.form.get("user_id")
    movie_id = request.form.get("movie_id")
    rating = request.form.get("rating")

    # Validate inputs
    if not user_id.isdigit() or not movie_id.isdigit() or not (1 <= float(rating) <= 5):
        flash("Invalid input values!", "error")
        return redirect(url_for("index"))

    print(f"user_id: {user_id}, movie_id: {movie_id}, rating: {rating}")  # Debug output

    try:
        model, ratings_df, n_users = update_model_with_new_rating(
            model,
            ratings_df,
            int(user_id),
            int(movie_id),
            float(rating),
            n_users,
            le_user,
            le_item,
        )
        flash("Rating added successfully!", "success")
    except Exception as e:
        flash(f"An error occurred: {str(e)}", "error")  # Capture the error message

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
