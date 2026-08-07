import os
import pickle
import numpy as np
import gradio as gr

# Resolve model.pkl relative to this script's own folder, so the app works
# regardless of the directory you launch it from.
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "model.pkl"
)

# ---- Feature configuration ----
# The model has no stored feature names, so generic labels are used below
# (Feature 1 ... Feature 12). Rename FEATURE_NAMES to match your actual
# dataset columns for a friendlier UI.
FEATURE_NAMES = [f"Feature {i + 1}" for i in range(12)]

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def predict(*values):
    X = np.array(values, dtype=float).reshape(1, -1)
    prediction = model.predict(X)[0]
    return f"{prediction:.4f}"


with gr.Blocks(title="Model Predictor") as demo:
    gr.Markdown("# 🌲 Random Forest Regressor")
    gr.Markdown(
        "Enter the input values below and click **Predict** to get the model's output."
    )

    with gr.Row():
        inputs = []
        for i, name in enumerate(FEATURE_NAMES):
            inputs.append(gr.Number(label=name, value=0.0))

    predict_btn = gr.Button("Predict", variant="primary")
    output = gr.Textbox(label="Predicted value")

    predict_btn.click(fn=predict, inputs=inputs, outputs=output)

    gr.Markdown(
        "---\n"
        "This app loads a pre-trained `RandomForestRegressor` "
        "(scikit-learn, 300 trees) from `models/model.pkl` and uses it to "
        "make predictions from 12 numeric input features. Update "
        "`FEATURE_NAMES` in `app.py` to reflect the real names of your "
        "features."
    )

if __name__ == "__main__":
    demo.launch()