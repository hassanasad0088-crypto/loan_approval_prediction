import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")


@st.cache_resource
def get_model():
    df = pd.read_csv("loan_prediction_train.csv")
    df = df.drop("Loan_ID", axis=1)

    for col in ["Gender", "Married", "Dependents", "Self_Employed",
                "Credit_History", "Loan_Amount_Term"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Dependents"] = df["Dependents"].replace("3+", "3").astype(int)
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
    df = pd.get_dummies(df, drop_first=True, dtype=int)

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)
    return model, scaler, list(X.columns)


model, scaler, columns = get_model()

st.title("🏦 Loan Approval Prediction")
st.write("Enter the applicant information to get the predicted loan approval probability and decision.")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0.0, value=5000.0, step=500.0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=500.0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=130.0, step=10.0)
    loan_term = st.number_input("Loan Term (months)", min_value=0.0, value=360.0, step=12.0)
    credit_history = st.selectbox("Credit History", [1, 0],
                                  format_func=lambda x: "1 (good)" if x == 1 else "0 (bad)")

if st.button("Predict", type="primary"):
    row = {
        "Dependents": int(dependents),
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": int(credit_history),
        "Gender_Male": int(gender == "Male"),
        "Married_Yes": int(married == "Yes"),
        "Education_Not Graduate": int(education == "Not Graduate"),
        "Self_Employed_Yes": int(self_employed == "Yes"),
        "Property_Area_Semiurban": int(property_area == "Semiurban"),
        "Property_Area_Urban": int(property_area == "Urban"),
    }
    data = pd.DataFrame([row])[columns]
    prob = model.predict_proba(scaler.transform(data))[0][1]

    st.subheader("Result")
    c1, c2 = st.columns(2)
    c1.metric("Approval probability", f"{prob * 100:.1f}%")
    c2.metric("Rejection probability", f"{(1 - prob) * 100:.1f}%")
    st.progress(float(prob))

    if prob >= 0.5:
        st.success("Prediction: APPROVED")
    else:
        st.error("Prediction: REJECTED")

st.caption("Model: Logistic Regression trained on the Loan Prediction dataset. For learning purposes only.")
