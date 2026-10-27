
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# Load the saved inventory model
MODEL_PATH = "best_inventory_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the saved model: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Inventory Risk Prediction",
    page_icon="📦",
    layout="wide"
)

# Application title
st.title("📦 Inventory Management System")
st.subheader("Inventory Stock Risk Prediction")

st.write(
    "Enter the product details below to predict its inventory stock status."
)

# Input section
st.header("Enter Inventory Details")

col1, col2 = st.columns(2)

with col1:
    current_stock = st.number_input(
        "Current Stock",
        min_value=0,
        value=10
    )

    avg_daily_demand = st.number_input(
        "Average Daily Demand",
        min_value=0.0,
        value=8.0
    )

    lead_time_days = st.number_input(
        "Lead Time (Days)",
        min_value=0,
        value=5
    )

    reorder_point = st.number_input(
        "Reorder Point",
        min_value=0,
        value=40
    )

    sales_last_30_days = st.number_input(
        "Sales in Last 30 Days",
        min_value=0,
        value=240
    )

with col2:
    stock_turnover_ratio = st.number_input(
        "Stock Turnover Ratio",
        min_value=0.0,
        value=3.5
    )

    forecast_error = st.number_input(
        "Forecast Error",
        value=0.15
    )

    season = st.selectbox(
        "Season",
        ["Winter", "Summer", "Spring", "Autumn"]
    )

    item_category = st.selectbox(
        "Item Category",
        ["Electronics", "Clothing", "Food", "Furniture"]
    )

    supplier_reliability = st.selectbox(
        "Supplier Reliability",
        ["High", "Medium", "Low"]
    )

# Prediction section
st.header("Predict Inventory Status")

if st.button("Predict Stock Status", type="primary"):

    # Build input using the same feature names used during training
    input_data = pd.DataFrame([{
        "current_stock": current_stock,
        "avg_daily_demand": avg_daily_demand,
        "lead_time_days": lead_time_days,
        "reorder_point": reorder_point,
        "sales_last_30_days": sales_last_30_days,
        "stock_turnover_ratio": stock_turnover_ratio,
        "forecast_error": forecast_error,
        "season": season,
        "item_category": item_category,
        "supplier_reliability": supplier_reliability
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.session_state["last_prediction"] = str(prediction)
        st.session_state["last_input"] = input_data.to_dict(
            orient="records"
        )[0]

        st.success(f"Predicted Stock Status: {prediction}")

        if str(prediction).lower() == "stockout":
            st.warning(
                "Stockout risk detected. Review stock levels "
                "and replenishment requirements."
            )
        elif str(prediction).lower() == "overstock":
            st.info(
                "Overstock risk detected. Review inventory levels "
                "and demand forecasts."
            )
        else:
            st.info(f"Model prediction: {prediction}")

        st.write("Input details:")
        st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# User Feedback Section
st.divider()
st.header("User Feedback")

if "last_prediction" not in st.session_state:
    st.info("Please make a prediction before submitting feedback.")
else:
    st.write(
        "Predicted Stock Status:",
        st.session_state["last_prediction"]
    )

    feedback = st.radio(
        "Was this prediction useful?",
        ["Yes", "No", "Needs Review"]
    )

    comments = st.text_area(
        "Additional Comments (Optional)"
    )

    if st.button("Submit Feedback"):
        feedback_file = Path("feedback.csv")

        feedback_record = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(),
            "predicted_status": st.session_state["last_prediction"],
            "feedback": feedback,
            "comments": comments
        }])

        try:
            if feedback_file.exists():
                feedback_record.to_csv(
                    feedback_file,
                    mode="a",
                    header=False,
                    index=False
                )
            else:
                feedback_record.to_csv(
                    feedback_file,
                    index=False
                )

            st.success("Feedback submitted successfully!")

        except Exception as e:
            st.error(f"Could not save feedback: {e}")
