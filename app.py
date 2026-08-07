import os
import pickle
import numpy as np
import streamlit as st

st.set_page_config(page_title="Model Predictor", page_icon="🌲", layout="centered")

# Resolve model.pkl relative to this script's own folder, so the app works
# regardless of the directory you launch `streamlit run` from.
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "model.pkl"
)

# ---- Feature configuration ----
# The uploaded model has no stored feature names, so generic labels are used
# below (Feature 1 ... Feature 12). Rename FEATURE_NAMES to match your
# actual dataset columns for a friendlier UI.
FEATURE_NAMES = [f"Feature {i+1}" for i in range(12)]


@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


model = load_model(MODEL_PATH)

st.title("🌲 Random Forest Regressor")
st.caption("Enter the input values below and click Predict to get the model's output.")

st.divider()

st.subheader("Input features")

cols = st.columns(2)
inputs = []
for i, name in enumerate(FEATURE_NAMES):
    with cols[i % 2]:
        val = st.number_input(name, value=0.0, format="%.4f", key=f"feat_{i}")
        inputs.append(val)

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    X = np.array(inputs).reshape(1, -1)
    prediction = model.predict(X)[0]
    st.success(f"### Predicted value: `{prediction:.4f}`")

with st.expander("About this app"):
    st.write(
        "This app loads a pre-trained `RandomForestRegressor` "
        "(scikit-learn, 300 trees) from `model.pkl` and uses it to make "
        "predictions from 12 numeric input features. Update `FEATURE_NAMES` "
        "in `app.py` to reflect the real names of your features."
    )