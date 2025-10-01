import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. Load model + training columns
# -----------------------------
@st.cache_resource
def load_model(path="model_with_columns.pkl"):
    loaded_obj = pickle.load(open(path, "rb"))
    
    if isinstance(loaded_obj, tuple):
        model, training_columns = loaded_obj
    elif isinstance(loaded_obj, list):
        model, training_columns = loaded_obj[0], None
    else:
        model, training_columns = loaded_obj, None

    return model, training_columns

model, training_columns = load_model()
st.write("Loaded model type:", type(model))
st.write("Training columns:", training_columns)

# -----------------------------
# 2. Upload CSV
# -----------------------------
st.title("🔮 Robust Clustering App with Feature Alignment")
uploaded_file = st.file_uploader("Upload CSV for clustering", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    # -----------------------------
    # 3. Preprocess uploaded data
    # -----------------------------
    processed_df = df.copy()

    # Drop ID / URL columns
    for col in processed_df.columns:
        if "id" in col.lower() or "link" in col.lower() or "url" in col.lower():
            processed_df.drop(columns=[col], inplace=True)

    # Encode categorical columns
    for col in processed_df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        processed_df[col] = le.fit_transform(processed_df[col].astype(str))

    # -----------------------------
    # 4. Align features with training columns
    # -----------------------------
    if training_columns:
        for col in training_columns:
            if col not in processed_df.columns:
                # Add missing columns filled with 0
                processed_df[col] = 0
        # Keep only training columns and correct order
        processed_df = processed_df[training_columns]

    X_input = processed_df.values

    # -----------------------------
    # 5. Predict / Cluster safely
    # -----------------------------
    try:
        if hasattr(model, "predict"):
            labels = model.predict(X_input)
        elif hasattr(model, "fit_predict"):
            labels = model.fit_predict(X_input)
        else:
            st.error("Model has neither predict nor fit_predict!")
            labels = None

        if labels is not None:
            processed_df["Cluster"] = labels
            st.write("### Clustered Data")
            st.dataframe(processed_df.head())

            # -----------------------------
            # 6. Visualization
            # -----------------------------
            if X_input.shape[1] >= 2:
                plt.figure(figsize=(8,6))
                plt.scatter(X_input[:,0], X_input[:,1], c=labels, cmap="viridis", s=30)

                if hasattr(model, "cluster_centers_"):
                    centers = model.cluster_centers_
                    plt.scatter(centers[:,0], centers[:,1], c="red", marker="X", s=200, label="Centers")
                    plt.legend()

                plt.title("Cluster Visualization")
                st.pyplot(plt)

            # -----------------------------
            # 7. Download clustered CSV
            # -----------------------------
            csv = processed_df.to_csv(index=False).encode()
            st.download_button(
                label="Download Clustered CSV",
                data=csv,
                file_name="clustered_data.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error during clustering: {e}")
