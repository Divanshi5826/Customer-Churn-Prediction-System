import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "treselle-systems/customer_churn_analysis/"
    "master/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "churn_model.pkl"
)


@st.cache_data
def load_data():

    data = pd.read_csv(DATA_URL)

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    data.dropna(inplace=True)

    return data


@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Customer Churn Analytics Dashboard")

st.markdown(
    """
    Analyze customer behavior, explore churn patterns,
    and predict the likelihood of customer churn.
    """
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Dashboard Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Churn Analysis",
        "Churn Prediction",
        "Model Performance"
    ]
)


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

if page == "Overview":

    st.header("Business Overview")

    total_customers = len(df)

    churned_customers = (
        df["Churn"] == "Yes"
    ).sum()

    churn_rate = (
        churned_customers / total_customers
    ) * 100

    avg_monthly_charges = (
        df["MonthlyCharges"].mean()
    )

    avg_tenure = (
        df["tenure"].mean()
    )


    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    col4.metric(
        "Avg Monthly Charges",
        f"${avg_monthly_charges:.2f}"
    )

    col5.metric(
        "Avg Tenure",
        f"{avg_tenure:.1f} months"
    )


    st.divider()

    st.subheader("Customer Churn Distribution")

    churn_counts = (
        df["Churn"]
        .value_counts()
    )

    st.bar_chart(churn_counts)


# --------------------------------------------------
# CHURN ANALYSIS
# --------------------------------------------------

elif page == "Churn Analysis":

    st.header("Customer Churn Analysis")

    analysis_option = st.selectbox(
        "Select Analysis",
        [
            "Contract Type",
            "Payment Method",
            "Internet Service",
            "Tenure",
            "Monthly Charges"
        ]
    )


    if analysis_option == "Contract Type":

        churn_data = pd.crosstab(
            df["Contract"],
            df["Churn"],
            normalize="index"
        ) * 100

        st.subheader(
            "Churn Rate by Contract Type"
        )

        st.bar_chart(
            churn_data["Yes"]
        )


    elif analysis_option == "Payment Method":

        churn_data = pd.crosstab(
            df["PaymentMethod"],
            df["Churn"],
            normalize="index"
        ) * 100

        st.subheader(
            "Churn Rate by Payment Method"
        )

        st.bar_chart(
            churn_data["Yes"]
        )


    elif analysis_option == "Internet Service":

        churn_data = pd.crosstab(
            df["InternetService"],
            df["Churn"],
            normalize="index"
        ) * 100

        st.subheader(
            "Churn Rate by Internet Service"
        )

        st.bar_chart(
            churn_data["Yes"]
        )


    elif analysis_option == "Tenure":

        st.subheader(
            "Tenure Distribution"
        )

        st.line_chart(
            df.groupby("tenure")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
        )


    elif analysis_option == "Monthly Charges":

        st.subheader(
            "Monthly Charges Distribution"
        )

        st.bar_chart(
            df.groupby(
                pd.cut(
                    df["MonthlyCharges"],
                    bins=10
                )
            )["Churn"]
            .apply(
                lambda x:
                (x == "Yes").mean() * 100
            )
        )


    st.divider()

    st.subheader("Key Business Insights")

    st.markdown(
        """
        - Month-to-month customers generally show higher churn
          than customers on longer-term contracts.

        - Customers with shorter tenure tend to have higher churn.

        - Churn behavior varies across payment methods and
          internet service categories.

        - Higher monthly charges are associated with increased
          churn in several customer segments.

        These insights can help businesses identify
        high-risk customer groups and design targeted
        retention strategies.
        """
    )


# --------------------------------------------------
# CHURN PREDICTION
# --------------------------------------------------

elif page == "Churn Prediction":

    st.header("🔮 Customer Churn Prediction")

    st.write(
        "Enter customer information to estimate "
        "the likelihood of churn."
    )


    col1, col2 = st.columns(2)


    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.slider(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )


    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=monthly_charges * tenure
        )


    if st.button(
        "Predict Churn",
        type="primary"
    ):

        input_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        })


        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]


        st.divider()

        if prediction == 1:

            st.error(
                f"⚠️ High Churn Risk — "
                f"{probability * 100:.2f}% probability"
            )

            st.warning(
                "This customer may require "
                "targeted retention efforts."
            )

        else:

            st.success(
                f"✅ Low Churn Risk — "
                f"{probability * 100:.2f}% probability"
            )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Model Performance")

    st.write(
        "Performance comparison of the machine learning models."
    )

    st.subheader("Models")

    st.markdown(
        """
        **Logistic Regression**

        Used as a baseline classification model.

        **Random Forest**

        Used as the final predictive model because
        of its ability to capture nonlinear relationships
        between customer attributes and churn behavior.
        """
    )

    st.info(
        "The exact performance metrics shown below "
        "are generated when the notebook is executed."
    )

    st.subheader("Project Methodology")

    st.markdown(
        """
        **1. Data Collection**

        Telco customer dataset containing demographic,
        subscription and billing information.

        **2. Data Cleaning**

        Converted TotalCharges to numeric format and
        handled invalid/missing values.

        **3. Exploratory Data Analysis**

        Analyzed churn distribution and customer behavior
        across contract, payment method, tenure,
        internet service and charges.

        **4. Data Preprocessing**

        Applied scaling to numerical variables and
        one-hot encoding to categorical variables.

        **5. Model Training**

        Trained Logistic Regression and Random Forest models.

        **6. Model Evaluation**

        Compared models using accuracy, precision,
        recall and F1-score.

        **7. Prediction**

        Built an interactive interface for estimating
        individual customer churn risk.
        """
    )