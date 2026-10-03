from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="Heart Disease Explorer",
    page_icon="❤️",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "final_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv(ROOT / "data" / "heart_disease.csv")


model = load_model()
df = load_data()

st.title("❤️ Heart Disease Explorer")
st.caption(
    "An educational machine-learning demonstration using the UCI dataset. "
    "Predictions are not a medical diagnosis or an estimate of future risk."
)

prediction_tab, exploration_tab = st.tabs(["Prediction", "Explore the dataset"])


# Prediction
with prediction_tab:
    st.subheader("Enter health measurements")
    st.caption(
        "Some inputs require clinical test results. "
        "The defaults are example values."
    )

    with st.form("health_inputs"):
        left, right = st.columns(2)

        with left:
            age = st.number_input(
                "Age (years)",
                min_value=1,
                max_value=120,
                value=55,
            )

            sex = st.selectbox(
                "Sex",
                [0, 1],
                format_func=lambda value: {0: "Female", 1: "Male"}[value],
            )

            cp = st.selectbox(
                "Chest pain type",
                [1, 2, 3, 4],
                format_func=lambda value: {
                    1: "Typical angina",
                    2: "Atypical angina",
                    3: "Non-anginal pain",
                    4: "Asymptomatic",
                }[value],
            )

            trestbps = st.number_input(
                "Resting blood pressure (mm Hg)",
                min_value=50,
                max_value=300,
                value=130,
            )

            chol = st.number_input(
                "Serum cholesterol (mg/dL)",
                min_value=50,
                max_value=700,
                value=240,
            )

            fbs = st.selectbox(
                "Fasting blood sugar above 120 mg/dL?",
                [0, 1],
                format_func=lambda value: "Yes" if value else "No",
            )

            restecg = st.selectbox(
                "Resting ECG result",
                [0, 1, 2],
                format_func=lambda value: {
                    0: "Normal",
                    1: "ST-T wave abnormality",
                    2: (
                        "Probable or definite left ventricular "
                        "hypertrophy"
                    ),
                }[value],
            )

        with right:
            thalach = st.number_input(
                "Maximum heart rate achieved (beats/min)",
                min_value=30,
                max_value=250,
                value=150,
            )

            exang = st.selectbox(
                "Exercise-induced angina?",
                [0, 1],
                format_func=lambda value: "Yes" if value else "No",
            )

            oldpeak = st.number_input(
                "Exercise-induced ST depression (oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
            )

            slope = st.selectbox(
                "Slope of peak exercise ST segment",
                [1, 2, 3],
                format_func=lambda value: {
                    1: "Upsloping",
                    2: "Flat",
                    3: "Downsloping",
                }[value],
            )

            ca = st.selectbox(
                "Number of major vessels colored by fluoroscopy",
                [0, 1, 2, 3],
            )

            thal = st.selectbox(
                "Thallium stress-test result",
                [3, 6, 7],
                format_func=lambda value: {
                    3: "Normal",
                    6: "Fixed defect",
                    7: "Reversible defect",
                }[value],
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        patient = pd.DataFrame(
            [{
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal,
            }]
        )

        # The saved pipeline does all preprocessing
        prediction = int(model.predict(patient)[0])

        if prediction == 1:
            st.info("Model prediction: disease present (class 1).")
        else:
            st.info("Model prediction: disease absent (class 0).")

        st.caption(f"Classifier: {type(model.named_steps['model']).__name__}")

        with st.expander("View submitted measurements"):
            st.dataframe(patient, hide_index=True)


# Dataset exploration
with exploration_tab:
    st.subheader("Patterns in the UCI dataset")
    st.caption(
        "These charts describe the dataset; associations do not establish "
        "causes."
    )

    explore_df = df.copy()
    explore_df["Disease"] = (explore_df["num"] > 0).map({
        False: "No disease",
        True: "Disease",
    })

    st.write(f"Patient records: **{len(explore_df)}**")

    counts = explore_df["Disease"].value_counts().reindex(
        ["No disease", "Disease"],
        fill_value=0,
    )

    st.write("Disease-label distribution")
    st.bar_chart(counts)

    feature_labels = {
        "age": "Age (years)",
        "trestbps": "Resting blood pressure (mm Hg)",
        "chol": "Cholesterol (mg/dL)",
        "thalach": "Maximum heart rate (beats/min)",
        "oldpeak": "ST depression",
        "ca": "Number of major vessels",
    }

    feature = st.selectbox(
        "Choose a feature to explore",
        list(feature_labels),
        format_func=lambda value: feature_labels[value],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(data=explore_df, x=feature, hue="Disease", bins=15, ax=axes[0])
    axes[0].set_title("Feature distribution")
    axes[0].set_xlabel(feature_labels[feature])

    sns.boxplot(
        data=explore_df,
        x="Disease",
        y=feature,
        order=["No disease", "Disease"],
        ax=axes[1],
    )
    axes[1].set_title("Comparison by disease label")
    axes[1].set_ylabel(feature_labels[feature])

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
