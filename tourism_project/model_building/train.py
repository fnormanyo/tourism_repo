
#import os for creating the deployment directory when it does not exist.
import os

#import pandas for reading and handling CSV datasets.
import pandas as pd

#import joblib for serializing the final trained model pipeline.
import joblib

#import XGBoost for the gradient-boosted binary classification model.
import xgboost as xgb

#import MLflow for experiment tracking, metric logging, model logging, and artifacts.
import mlflow

#import MLflow's scikit-learn flavor to log the fitted sklearn pipeline as an MLflow model.
import mlflow.sklearn

#import transformations for numeric and categorical feature preprocessing.
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#import a transformer for applying different preprocessing to separate column groups.
from sklearn.compose import make_column_transformer

#import a utility for combining preprocessing and modeling into a single pipeline.
from sklearn.pipeline import make_pipeline

#import GridSearchCV for cross-validated hyperparameter tuning.
from sklearn.model_selection import GridSearchCV

#import classification_report for calculating classification performance metrics.
from sklearn.metrics import classification_report


#set the MLflow tracking server URI.
mlflow.set_tracking_uri("http://localhost:5000")

#set or create the MLflow experiment where all runs will be stored.
mlflow.set_experiment("Tourism Package Purchase Prediction")


#load training feature data created by the upstream data-preparation job.
Xtrain = pd.read_csv("Xtrain.csv")

#load test feature data created by the upstream data-preparation job.
Xtest = pd.read_csv("Xtest.csv")

#load and flatten the training target values into a one-dimensional Series.
ytrain = pd.read_csv("ytrain.csv").squeeze()

#load and flatten the test target values into a one-dimensional Series.
ytest = pd.read_csv("ytest.csv").squeeze()


#define numeric columns that should be standardized before training.
numeric_features = [
    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch",
]


#define categorical columns that should be converted into one-hot encoded features.
categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched",
]


#count records in each target class for class-imbalance handling.
class_counts = ytrain.value_counts()

#calculate the negative-to-positive class ratio required by XGBoost scale_pos_weight.
class_weight = class_counts[0] / class_counts[1]


#standardize numeric features and one-hot encode categorical features.
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)


#create the base XGBoost binary classifier with class-imbalance adjustment.
model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
)


#define a compact grid to keep GitHub Actions training time reasonable.
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

#combine preprocessing and XGBoost classification in one fitted pipeline.
pipeline = make_pipeline(preprocessor, model)

#define the classification probability cutoff used for positive predictions.
classification_threshold = 0.45

#define the local deployment location used by the Streamlit application.
m_path = "tourism_project/deployment/best_tourism_model_v0.joblib"

#create the deployment directory before saving the trained model.
os.makedirs(os.path.dirname(m_path), exist_ok=True)


#start the parent MLflow run for the complete training workflow.
with mlflow.start_run(run_name="xgboost_grid_search"):

    #log fixed experiment settings that affect model behavior and evaluation.
    mlflow.log_params({
        "cv_folds": 5,
        "scoring_metric": "recall",
        "classification_threshold": classification_threshold,
        "scale_pos_weight": class_weight,
        "random_state": 42,
    })

    #configure five-fold cross-validation to optimize recall for the positive class.
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1,
        return_train_score=False,
    )

    #fit all hyperparameter combinations and select the configuration with highest CV recall.
    grid.fit(Xtrain, ytrain)

    #access detailed results from every hyperparameter configuration tested.
    results = grid.cv_results_

    #log a nested MLflow run for each GridSearchCV hyperparameter combination.
    for i, param_set in enumerate(results["params"]):

        #retrieve mean validation recall for the current parameter combination.
        mean_score = results["mean_test_score"][i]

        #retrieve standard deviation of validation recall across the CV folds.
        std_score = results["std_test_score"][i]

        #retrieve the ranking of this parameter set among all tested combinations.
        rank = results["rank_test_score"][i]

        #start a child run associated with the parent model-training run.
        with mlflow.start_run(run_name=f"grid_search_trial_{i + 1}", nested=True):

            #log the hyperparameters tested in this child run.
            mlflow.log_params(param_set)

            #log validation metrics and ranking produced by GridSearchCV.
            mlflow.log_metrics({
                "mean_cv_recall": float(mean_score),
                "std_cv_recall": float(std_score),
                "cv_rank": int(rank),
            })

    #retrieve the final fitted pipeline with the best hyperparameter values.
    best_model = grid.best_estimator_

    #log only the best hyperparameters in the parent MLflow run.
    mlflow.log_params(grid.best_params_)

    #log the strongest average cross-validation recall produced during tuning.
    mlflow.log_metric("best_cv_recall", float(grid.best_score_))

    #display the selected configuration in GitHub Actions or terminal logs.
    print("Best params:", grid.best_params_)
    print(f"Best cross-validation recall: {grid.best_score_:.4f}")

    #generate positive-class probabilities for the training data.
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]

    #convert training probabilities to class labels using the defined cutoff.
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    #generate positive-class probabilities for the held-out test data.
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

    #convert test probabilities to class labels using the defined cutoff.
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    #calculate training metrics while safely handling undefined precision or recall values.
    train_report = classification_report(
        ytrain,
        y_pred_train,
        output_dict=True,
        zero_division=0,
    )

    #calculate held-out test metrics while safely handling undefined precision or recall values.
    test_report = classification_report(
        ytest,
        y_pred_test,
        output_dict=True,
        zero_division=0,
    )

    #log final performance metrics from the best fitted model.
    mlflow.log_metrics({
        "train_accuracy": float(train_report["accuracy"]),
        "train_precision": float(train_report["1"]["precision"]),
        "train_recall": float(train_report["1"]["recall"]),
        "train_f1_score": float(train_report["1"]["f1-score"]),
        "test_accuracy": float(test_report["accuracy"]),
        "test_precision": float(test_report["1"]["precision"]),
        "test_recall": float(test_report["1"]["recall"]),
        "test_f1_score": float(test_report["1"]["f1-score"]),
    })

    #save the full preprocessing-and-classifier pipeline for direct Streamlit inference.
    joblib.dump(best_model, m_path)

    #log the Joblib model file as an artifact of this active parent MLflow run.
    mlflow.log_artifact(m_path, artifact_path="model")

    #log the model in MLflow's native scikit-learn format for registry and reproducible loading.
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="sklearn_model",
    )

    #print the model location used by the deployment application.
    print(f"Model saved to {m_path}")
