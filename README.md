# Amazon-Music-Clustering
A simple Streamlit app that loads a saved clustering model (or estimator) and applies it to an uploaded CSV file.
The app performs light preprocessing (drops id/url columns, label-encodes object columns), aligns uploaded features to the training columns saved with the model, runs the model's predict / fit_predict, visualizes clusters (first two features), and provides a downloadable clustered CSV.

Features:

Upload any CSV and automatically preprocess it (drop id/url-like columns, label-encode object columns).

Align uploaded dataset to the training columns the model was trained with (if provided).

Supports models that expose predict or fit_predict.

2D scatter visualization (first two features) with cluster centers (if available).

Download the clustered CSV.

Files:

dash.py — main Streamlit app (the file you provided). 

dash

model_with_columns.pkl — expected pickled model file (place it in the same directory or pass the path in code).

Requirements:

Python 3.8+ (recommended)

streamlit

pandas

numpy

scikit-learn

matplotlib

(optional) whichever libraries your saved model depends on

Install with pip:

pip install streamlit pandas numpy scikit-learn matplotlib

Installation

Clone or copy this repository to your machine.

Put the saved model file model_with_columns.pkl into the project root (see format below).

Saved model format

dash.py expects a pickle file named model_with_columns.pkl by default. The loader handles a few shapes:

tuple: (model, training_columns)

model is any object with .predict(X) or .fit_predict(X) or .cluster_centers_ (for visualization).

training_columns is a list of the column names (in correct order) used during training.

list: the code expects a list and extracts the first element as the model (but then has no training_columns saved).

or a raw model object.

Recommended format: save a tuple (model, training_columns) so the app can align uploaded data automatically.

CSV input expectations

CSV should contain feature columns used during model training.

Columns with "id", "url", or "link" in their name are dropped automatically.

Any object (string) columns are label-encoded before prediction.

If your uploaded CSV lacks training columns, the app will add missing columns filled with zeros (but results may be meaningless if crucial features are missing).

Notes & tips

The app label-encodes every object column using a fresh LabelEncoder. If your model was trained with a particular label mapping, you should incorporate the encoding into the saved pipeline (so that at inference time labels match training mapping). Best practice: build and save a pipeline that contains preprocessing and model (so you don't need to re-implement encoders at inference).

For reliable results, save a sklearn.pipeline.Pipeline with both preprocessing and estimator; then pickling the pipeline alone is sufficient.

The app visualizes the first two numeric features as X and Y. If your features are not directly comparable or require scaling, include scaling in your saved pipeline.

The app caches model loading with @st.cache_resource. If you replace the model_with_columns.pkl while Streamlit is running, restart Streamlit to reload the new model.

Contributing

If you'd like enhancements (e.g., custom preprocessing, consistent label encoders, selectable feature projections like PCA/t-SNE, or a preview of missing columns), open an issue or submit a PR.



