import streamlit as st
import pandas as pd
from src.recommender import load_data, train_model, recommend_for_user, evaluate_model, DATA_DIR, MODEL_DIR

st.set_page_config(page_title="MovieMind | MLP Movie Preferences", page_icon="🎬", layout="wide")

st.title("🎬 MovieMind")
st.subheader("Movie Preference Prediction using a Multilayer Perceptron (MLP)")
st.write("Explore MovieLens ratings, predict whether a user is likely to enjoy a movie, and discover personalized recommendations.")

with st.sidebar:
    st.header("⚙️ Project controls")
    st.write("Dataset: MovieLens Latest Small")
    st.caption("On first run, the app downloads the public dataset if it is not already available.")
    force_retrain = st.button("Retrain model")
    st.markdown("---")
    st.caption("Preference label used for training: rating ≥ 4.0 out of 5 = Like.")

@st.cache_resource(show_spinner="Loading data and training the MLP model…")
def get_resources():
    data = load_data()
    model, bundle, metrics = train_model(data)
    return data, model, bundle, metrics

try:
    if force_retrain:
        get_resources.clear()
    data, model, bundle, metrics = get_resources()
except Exception as exc:
    st.error("The dataset or model could not be loaded.")
    st.exception(exc)
    st.info("Check your internet connection on first run, then run: python -m pip install -r requirements.txt")
    st.stop()

movies = data["movies"]
ratings = data["ratings"]
user_ids = sorted(ratings["userId"].unique().tolist())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Movies", f"{len(movies):,}")
c2.metric("Ratings", f"{len(ratings):,}")
c3.metric("Users", f"{ratings['userId'].nunique():,}")
c4.metric("Test accuracy", f"{metrics['accuracy']:.1%}")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["🎯 Get recommendations", "🔍 Predict a movie", "📊 Model evaluation"])

with tab1:
    st.header("Personalized recommendations")
    st.write("Choose a sample MovieLens user. Recommendations are based on that user's rating history and the MLP's predicted like probability.")
    user_id = st.selectbox("Choose user ID", user_ids, index=0, key="rec_user")
    count = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)
    if st.button("Recommend movies", type="primary"):
        recs = recommend_for_user(data, model, bundle, int(user_id), n=count)
        if recs.empty:
            st.warning("No unrated movies were available for this user.")
        else:
            show = recs[["title", "genres", "predicted_like_probability", "average_rating"]].copy()
            show["predicted_like_probability"] = (show["predicted_like_probability"] * 100).round(1).astype(str) + "%"
            show["average_rating"] = show["average_rating"].round(2)
            show.columns = ["Movie", "Genres", "Predicted chance of Like", "MovieLens average rating / 5"]
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.download_button("Download recommendations as CSV", recs.to_csv(index=False).encode("utf-8"), file_name=f"recommendations_user_{user_id}.csv", mime="text/csv")

with tab2:
    st.header("Predict preference for a movie")
    left, right = st.columns([2, 1])
    with left:
        selected_user = st.selectbox("User", user_ids, index=0, key="pred_user")
        movie_options = movies.sort_values("title")
        selected_title = st.selectbox("Movie", movie_options["title"].tolist(), key="pred_movie")
    with right:
        st.write("")
        st.write("")
        if st.button("Predict preference", type="primary"):
            row = movies.loc[movies["title"] == selected_title].iloc[0]
            from src.recommender import predict_movie_for_user
            result = predict_movie_for_user(data, model, bundle, int(selected_user), int(row["movieId"]))
            if result is None:
                st.warning("The selected movie could not be scored.")
            else:
                probability = result["probability"]
                if probability >= 0.5:
                    st.success(f"Likely to LIKE — {probability:.1%}")
                else:
                    st.info(f"Less likely to LIKE — {probability:.1%}")
                st.progress(float(max(0, min(1, probability))))
                st.caption("This is a model estimate based on historical MovieLens patterns, not a guarantee of a person's actual opinion.")

with tab3:
    st.header("MLP model evaluation")
    st.write("The model is evaluated on a held-out test split. A Like is defined as a rating of 4.0 or above.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("Precision", f"{metrics['precision']:.1%}")
    m3.metric("Recall", f"{metrics['recall']:.1%}")
    m4.metric("F1 score", f"{metrics['f1']:.1%}")
    st.write("Confusion matrix (rows = actual, columns = predicted)")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=["Actual Dislike", "Actual Like"], columns=["Predicted Dislike", "Predicted Like"]), use_container_width=True)
    st.write("Model architecture")
    st.code("Input features → Dense 64 → ReLU → Dense 32 → ReLU → Like / Dislike\nImplemented with scikit-learn MLPClassifier(hidden_layer_sizes=(64, 32))")
    st.caption("Accuracy can vary slightly with library versions and dataset contents. The model uses movie genres, a user's genre preferences, and rating-history summaries.")
