import os
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="✈️",
    layout="centered",
)


@st.cache_resource
def load_model(model_file_path):
    """Load the trained model once per Streamlit server process."""
    loaded_object = joblib.load(model_file_path)

    # Supports either:
    # 1. joblib.dump(model, ...)
    # 2. joblib.dump({"model": model, "features": [...]}, ...)
    if isinstance(loaded_object, dict) and "model" in loaded_object:
        return loaded_object["model"], loaded_object.get("features")

    return loaded_object, None


def get_expected_features(model, bundled_features=None):
    """
    Return the raw DataFrame feature names required by the fitted pipeline.

    Preference order:
    1. Explicit schema saved with the model bundle.
    2. Pipeline/model feature_names_in_.
    3. Feature names stored on a ColumnTransformer within named_steps.
    """
    if bundled_features:
        return list(bundled_features)

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


def prepare_input_data(form_values, expected_features):
    """
    Build a one-row DataFrame and validate the deployment schema.

    A missing model feature indicates that the Streamlit form and the
    fitted pipeline are out of sync. Extra form columns are removed.
    """
    input_df = pd.DataFrame([form_values])

    if expected_features is None:
        return input_df, [], []

    missing_columns = sorted(set(expected_features) - set(input_df.columns))
    extra_columns = sorted(set(input_df.columns) - set(expected_features))

    # Reindex preserves the exact feature order used by the fitted model.
    # Missing columns remain visible through missing_columns and are not
    # silently filled with an arbitrary value.
    input_df = input_df.reindex(columns=expected_features)

    return input_df, missing_columns, extra_columns


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_tourism_model_v1.joblib",
)

try:
    model, bundled_features = load_model(MODEL_PATH)
    expected_features = get_expected_features(model, bundled_features)
except FileNotFoundError:
    st.error(
        "The trained model file was not found. "
        "Confirm that `best_tourism_model_v1.joblib` is in the deployment folder."
    )
    st.stop()
except Exception:
    st.error("The trained model could not be loaded. Check the Streamlit logs.")
    st.stop()


st.title("✈️ Tourism Package Purchase Predictor")
st.markdown(
    "This application predicts whether a customer is likely to purchase "
    "the Wellness Tourism Package."
)

st.header("Customer Information")

age = st.slider("Age", min_value=18, max_value=80, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
city_tier = st.selectbox("City Tier", [1, 2, 3])

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"],
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0,
    value=50000,
    step=1000,
)

st.header("Travel and Contact Details")

type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Inquiry", "Company Invited"],
)

number_of_person_visiting = st.slider(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=1,
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5],
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced", "Unmarried"],
)

number_of_trips = st.slider(
    "Number of Trips Annually",
    min_value=0,
    max_value=20,
    value=1,
)

passport = st.selectbox(
    "Passport",
    [0, 1],
    format_func=lambda value: "Yes" if value == 1 else "No",
)

own_car = st.selectbox(
    "Own Car",
    [0, 1],
    format_func=lambda value: "Yes" if value == 1 else "No",
)

number_of_children_visiting = st.slider(
    "Number of Children Visiting (under 5)",
    min_value=0,
    max_value=5,
    value=0,
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Director"],
)

st.header("Sales Pitch Information")

pitch_satisfaction_score = st.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3,
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
)

number_of_followups = st.slider(
    "Number of Follow-ups",
    min_value=0,
    max_value=10,
    value=2,
)

duration_of_pitch = st.slider(
    "Duration of Pitch (minutes)",
    min_value=1,
    max_value=60,
    value=10,
)


form_values = {
    "Age": age,
    "Gender": gender,
    "CityTier": city_tier,
    "Occupation": occupation,
    "MonthlyIncome": monthly_income,
    "TypeofContact": type_of_contact,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": number_of_followups,
    "DurationOfPitch": duration_of_pitch,
}

input_data, missing_columns, extra_columns = prepare_input_data(
    form_values=form_values,
    expected_features=expected_features,
)


with st.sidebar:
    with st.expander("Developer diagnostics"):
        st.write("Expected model features:", expected_features)
        st.write("Application form features:", list(form_values.keys()))
        st.write("Missing model features:", missing_columns)
        st.write("Extra form features:", extra_columns)


if st.button("Predict Purchase", type="primary"):
    if missing_columns:
        st.error(
            "The application form does not match the feature schema used to "
            "train the deployed model. Missing fields: "
            + ", ".join(missing_columns)
        )
        st.info(
            "Add the missing fields with the exact training-column names, "
            "or deploy the matching model version."
        )
        st.stop()

    if not hasattr(model, "predict_proba"):
        st.error(
            "The loaded model does not support probability predictions. "
            "Use a classifier with `predict_proba`, or implement a "
            "decision-score conversion deliberately."
        )
        st.stop()

    try:
        prediction_proba = float(model.predict_proba(input_data)[0, 1])
        prediction = int(prediction_proba >= 0.50)
    except ValueError as error:
        st.error("Prediction could not be completed because of a model-input mismatch.")
        st.info(
            "Review the expected and supplied columns under "
            "`Developer diagnostics` in the sidebar."
        )
        st.exception(error)
        st.stop()
    except Exception:
        st.error("Prediction could not be completed. Review the Streamlit logs.")
        st.stop()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "The customer is likely to purchase the package "
            f"(predicted probability: {prediction_proba:.1%})."
        )
    else:
        st.warning(
            "The customer is unlikely to purchase the package "
            f"(predicted probability: {prediction_proba:.1%})."
        )

    st.caption("The default purchase classification threshold is 50%.")
