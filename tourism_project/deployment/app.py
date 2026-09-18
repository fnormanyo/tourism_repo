
#import the operating-system module for safe file-path construction.
import os

#import joblib to load the saved machine-learning model.
import joblib

#import pandas to create and validate prediction input data.
import pandas as pd

#import Streamlit to build the interactive web application.
import streamlit as st

#configure the Streamlit page title and centered application layout.
st.set_page_config(
    page_title="Tourism Package Predictor",
    layout="centered",
)

#cache the loaded model so it is not reloaded after every user interaction.
@st.cache_resource
def load_model(model_file_path):
    """Load the trained model once per Streamlit server process."""

    #load the serialized model object from the specified joblib file.
    loaded_object = joblib.load(model_file_path)

    #return the model and optional feature list when the file stores a dictionary bundle.
    if isinstance(loaded_object, dict) and "model" in loaded_object:
        return loaded_object["model"], loaded_object.get("features")

    #return a standalone model and indicate that no bundled feature list exists.
    return loaded_object, None


#identify the feature names expected by the trained model pipeline.
def get_expected_features(model, bundled_features=None):
    """Return the raw DataFrame feature names required by the fitted pipeline."""

    #use the explicit feature list saved together with the model when available.
    if bundled_features:
        return list(bundled_features)

    #retrieve feature names directly from the model when the attribute exists.
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    #search each pipeline step for the feature names used during model training.
    if hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    #return None when the expected feature schema cannot be determined.
    return None


#create a one-row DataFrame and align it with the model's expected schema.
def prepare_input_data(form_values, expected_features):
    """
    Build a one-row DataFrame and validate the deployment schema.

    A missing model feature indicates that the Streamlit form and the
    fitted pipeline are out of sync. Extra form columns are removed.
    """

    #convert the submitted form values dictionary into a single-row DataFrame.
    input_df = pd.DataFrame([form_values])

    #return the raw input when the model does not expose expected feature names.
    if expected_features is None:
        return input_df, [], []

    #identify model-required columns that are absent from the form input.
    missing_columns = sorted(set(expected_features) - set(input_df.columns))

    #identify form columns that are not required by the deployed model.
    extra_columns = sorted(set(input_df.columns) - set(expected_features))

    #reorder columns to match the training schema and discard extra columns.
    input_df = input_df.reindex(columns=expected_features)

    #return the aligned input data along with schema-validation results.
    return input_df, missing_columns, extra_columns


#build the absolute path to the saved machine-learning model file.
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_tourism_model_v0.joblib",
)

#load the deployed model and determine its expected input features.
try:
    model, bundled_features = load_model(MODEL_PATH)
    expected_features = get_expected_features(model, bundled_features)

#display a clear error when the joblib model file cannot be found.
except FileNotFoundError:
    st.error(
        "The trained model file was not found. "
        "Confirm that `best_tourism_model_v0.joblib` is in the deployment folder."
    )
    st.stop()

#display a generic error when the model cannot be loaded for another reason.
except Exception:
    st.error("The trained model could not be loaded. Check the Streamlit logs.")
    st.stop()


#display the main application title.
st.title("Tourism Package Purchase Predictor")

#explain the prediction objective of the application.
st.markdown(
    "Application to predicts whether a customer is likely to purchase "
    "the Wellness Tourism Package."
)


#display the first form section for customer demographic details.
st.header("Customer Information")


#collect the customer's age using a bounded slider.
age = st.slider("Age", min_value=18, max_value=80, value=30)

#collect the customer's gender using a dropdown selection.
gender = st.selectbox("Gender", ["Male", "Female"])

#collect the customer's city tier classification.
city_tier = st.selectbox("City Tier", [1, 2, 3])


#collect the customer's occupation category.
occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"],
)


#collect the customer's monthly income value.
monthly_income = st.number_input(
    "Monthly Income",
    min_value=0,
    value=50000,
    step=1000,
)


#display the form section for travel history and contact information.
st.header("Travel and Contact Details")


#collect how the customer first contacted or was contacted by the company.
type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Inquiry", "Company Invited"],
)


#collect the total number of people expected to travel.
number_of_person_visiting = st.slider(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=1,
)


#collect the customer's preferred hotel or property star rating.
preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5],
)


#collect the customer's marital status.
marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced", "Unmarried"],
)


#collect the number of trips the customer takes per year.
number_of_trips = st.slider(
    "Number of Trips Annually",
    min_value=0,
    max_value=20,
    value=1,
)

#collect passport ownership as a binary model input.
passport = st.selectbox(
    "Passport",
    [0, 1],
    format_func=lambda value: "Yes" if value == 1 else "No",
)

#collect car ownership as a binary model input.
own_car = st.selectbox(
    "Own Car",
    [0, 1],
    format_func=lambda value: "Yes" if value == 1 else "No",
)


#collect the number of children under five expected to travel.
number_of_children_visiting = st.slider(
    "Number of Children Visiting (under 5)",
    min_value=0,
    max_value=5,
    value=0,
)


#collect the customer's job designation or seniority level.
designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Director"],
)


#display the form section for sales-interaction details.
st.header("Sales Pitch Information")


#collect the customer's satisfaction score for the sales pitch.
pitch_satisfaction_score = st.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3,
)


#collect the tourism package product pitched to the customer.
product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
)


#collect the number of sales follow-ups made to the customer.
number_of_followups = st.slider(
    "Number of Follow-ups",
    min_value=0,
    max_value=10,
    value=2,
)


#collect the duration of the tourism package sales pitch in minutes.
duration_of_pitch = st.slider(
    "Duration of Pitch (minutes)",
    min_value=1,
    max_value=60,
    value=10,
)


#map Streamlit widget values to the exact feature names used during training.
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


#validate and reorder the form data to match the model's expected feature schema.
input_data, missing_columns, extra_columns = prepare_input_data(
    form_values=form_values,
    expected_features=expected_features,
)


#display schema and feature diagnostics in the collapsible sidebar panel.
with st.sidebar:
    with st.expander("Developer diagnostics"):
        st.write("Expected model features:", expected_features)
        st.write("Application form features:", list(form_values.keys()))
        st.write("Missing model features:", missing_columns)
        st.write("Extra form features:", extra_columns)


#trigger model prediction only after the user clicks the primary prediction button.
if st.button("Predict Purchase", type="primary"):

    #stop prediction when the application is missing features required by the model.
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

    #confirm that the deployed classifier can produce probability estimates.
    if not hasattr(model, "predict_proba"):
        st.error(
            "Loaded model does not support probability predictions. "
            "Use a classifier with `predict_proba`, or implement a "
            "decision-score conversion deliberately."
        )
        st.stop()

    #generate a probability prediction and apply the 45% classification threshold.
    try:
        prediction_proba = float(model.predict_proba(input_data)[0, 1])
        prediction = int(prediction_proba >= 0.45)

    #handle input-format or feature-schema problems returned by the model.
    except ValueError as error:
        st.error("Prediction could not be completed because of a model-input mismatch.")
        st.info(
            "Review the expected and supplied columns under "
            "`Developer diagnostics` in the sidebar."
        )
        st.exception(error)
        st.stop()

    #handle unexpected errors that occur during prediction.
    except Exception:
        st.error("Prediction could not be completed. Review the Streamlit logs.")
        st.stop()

    #display the heading for the classification output.
    st.subheader("Prediction Result")

    #show a success message when the predicted purchase class is positive.
    if prediction == 1:
        st.success(
            "The customer is likely to purchase the package "
            f"(predicted probability: {prediction_proba:.1%})."
        )

    #show a warning message when the predicted purchase class is negative.
    else:
        st.warning(
            "The customer is unlikely to purchase the package "
            f"(predicted probability: {prediction_proba:.1%})."
        )

    #explain the default probability threshold used to determine the final class.
    st.caption("The default purchase classification threshold is 45%.")
