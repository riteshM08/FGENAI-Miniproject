# Movie Preference Prediction Using MLP

A complete beginner-friendly machine learning mini-project using MovieLens ratings, a Multilayer Perceptron (MLP), and Streamlit.

## What it does
- Downloads the public MovieLens Latest Small dataset on first run.
- Converts ratings into a binary target: rating >= 4.0 means Like; below 4.0 means Dislike.
- Builds numerical features from movie genres, user genre preferences, and rating-history summaries.
- Trains an MLP classifier and reports accuracy, precision, recall, F1 score, and a confusion matrix.
- Provides a Streamlit interface for recommendations and single-movie predictions.

## Requirements
Recommended: Python 3.11 or 3.12. Python 3.14 may not yet have compatible wheels for every scientific Python package in all environments.

## Windows PowerShell setup
Open PowerShell in this project folder.

```powershell
py -0p
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

If you already have a `venv` created with Python 3.14, don't reuse it for Python 3.12. Create the separate `venv` as above after removing/renaming the old one, or use a new name such as `venv312`.

## Run
```powershell
streamlit run app.py
```
The app opens in your browser. On first launch, internet access is needed to download MovieLens.

## Project structure
```text
app.py
src/
  __init__.py
  recommender.py
data/       # downloaded CSVs appear here
models/     # reserved for saved model artifacts
requirements.txt
README.md
```

## Notes and limitations
- MovieLens is a public research dataset. Its user IDs are anonymous dataset identifiers, not the website visitor's identity.
- The model estimates preference patterns; it cannot guarantee a real person's taste.
- This is a learning mini-project, not a production-grade recommendation system.
- A random rating-level split can make evaluation optimistic because the same users and movies can appear in train and test. For a more rigorous study, use a time-based or user-aware evaluation split.
- The first run requires internet access to download the dataset.
