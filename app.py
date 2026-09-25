
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="🏨",
    layout="wide"
)

# -----------------------------
# Load Trained Model
# -----------------------------
MODEL_PATH = "model_building/tourism_package_model.pkl"
model = joblib.load(MODEL_PATH)

# -----------------------------
# Title
# -----------------------------
st.title("🏨 Tourism Package Purchase Prediction")
st.write(
    "Enter the customer details below to predict whether the customer "
    "is likely to purchase the Wellness Tourism Package."
)

st.divider()

# -----------------------------
# Input Fields
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

    TypeofContact = st.selectbox(
        "Type of Contact",
        ["Company Invited", "Self Enquiry"]
    )

    CityTier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    DurationOfPitch = st.number_input(
        "Duration of Pitch (minutes)",
        min_value=0,
        max_value=60,
        value=15,
        step=1
    )

    Occupation = st.selectbox(
        "Occupation",
        ["Freelancer", "Large Business", "Salaried", "Small Business"]
    )

    Gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    NumberOfPersonVisiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )

    NumberOfFollowups = st.number_input(
        "Number of Followups",
        min_value=0,
        max_value=20,
        value=3,
        step=1
    )

    ProductPitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"]
    )

    PreferredPropertyStar = st.number_input(
        "Preferred Property Star",
        min_value=1,
        max_value=5,
        value=3,
        step=1
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        ["Divorced", "Married", "Single"]
    )

    NumberOfTrips = st.number_input(
        "Number of Trips",
        min_value=0,
        max_value=50,
        value=3,
        step=1
    )

with col3:
    Passport = st.selectbox(
        "Passport",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    PitchSatisfactionScore = st.number_input(
        "Pitch Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3,
        step=1
    )

    OwnCar = st.selectbox(
        "Own Car",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    NumberOfChildrenVisiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

    Designation = st.selectbox(
        "Designation",
        ["AVP", "Executive", "Manager", "Senior Manager", "VP"]
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=0,
        max_value=1000000,
        value=25000,
        step=1000
    )

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Package Purchase", type="primary"):

    input_data = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome
    }])

    # Ensure exact feature order used during training
    input_data = input_data[[
        "Age",
        "TypeofContact",
        "CityTier",
        "DurationOfPitch",
        "Occupation",
        "Gender",
        "NumberOfPersonVisiting",
        "NumberOfFollowups",
        "ProductPitched",
        "PreferredPropertyStar",
        "MaritalStatus",
        "NumberOfTrips",
        "Passport",
        "PitchSatisfactionScore",
        "OwnCar",
        "NumberOfChildrenVisiting",
        "Designation",
        "MonthlyIncome"
    ]]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("The customer is predicted to purchase the package.")
    else:
        st.info("The customer is predicted not to purchase the package.")

    st.metric(
        "Purchase Probability",
        f"{probability:.2%}"
    )
