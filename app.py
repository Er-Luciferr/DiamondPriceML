import os

import streamlit as st

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


def _artifacts_exist() -> bool:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    return os.path.isfile(os.path.join(base, "model.pkl")) and os.path.isfile(
        os.path.join(base, "preprocessor.pkl")
    )


st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered",
)

st.title("Diamond Price Prediction")
st.subheader("Regression model")
st.caption("Enter diamond features to estimate price (INR).")

if not _artifacts_exist():
    st.warning(
        "Trained model artifacts not found. Run training first from the project root:\n\n"
        "`python src/pipeline/training_pipeline.py`"
    )
    st.stop()

CUT_OPTIONS = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_OPTIONS = ["D", "E", "F", "G", "H", "I", "J"]
CLARITY_OPTIONS = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
PRICE_SCALE = 83.0

with st.form("diamond_form"):
    c1, c2 = st.columns(2)
    with c1:
        carat = st.number_input("Carat", min_value=0.0, value=0.5, step=0.01, format="%.4f")
        depth = st.number_input("Depth", min_value=0.0, value=60.0, step=0.1, format="%.2f")
        table = st.number_input("Table", min_value=0.0, value=55.0, step=0.1, format="%.2f")
        x = st.number_input("x (length)", min_value=0.0, value=5.0, step=0.01, format="%.4f")
    with c2:
        y = st.number_input("y (width)", min_value=0.0, value=5.0, step=0.01, format="%.4f")
        z = st.number_input("z (depth)", min_value=0.0, value=3.0, step=0.01, format="%.4f")
        cut = st.selectbox("Cut", CUT_OPTIONS, index=4)
        color = st.selectbox("Color", COLOR_OPTIONS, index=2)
        clarity = st.selectbox("Clarity", CLARITY_OPTIONS, index=3)

    submitted = st.form_submit_button("Predict price")

if submitted:
    try:
        data = CustomData(
            carat=float(carat),
            depth=float(depth),
            table=float(table),
            x=float(x),
            y=float(y),
            z=float(z),
            cut=cut,
            color=color,
            clarity=clarity,
        )
        df = data.get_data_as_dataframe()
        pipeline = PredictPipeline()
        pred = pipeline.predict(df)
        raw = round(float(pred[0]), 2)
        inr = raw * PRICE_SCALE
        st.success(f"**Estimated price: Rs. {inr:,.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
