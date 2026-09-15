
import pandas as pd
import joblib
import xgboost as xgb
import mlflow
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

mlflow.set_tracking_uri("http://localhost:5000")   # MLflow tracking URI
mlflow.set_experiment("Tourism Package Purchase Prediction")     # MLflow experiment name

#download training and test data from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()


numeric_features = ['Age',
                    'CityTier',
                    'NumberOfPersonVisiting',
                    'PreferredPropertyStar',
                    'NumberOfTrips',
                    'Passport',
                    'OwnCar',
                    'NumberOfChildrenVisiting',
                    'MonthlyIncome',
                    'PitchSatisfactionScore',
                    'NumberOfFollowups',
                    'DurationOfPitch']

categorical_features = ['TypeofContact',
                        'Occupation',
                        'Gender',
                        'MaritalStatus',
                        'Designation',
                        'ProductPitched']


#handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

#define preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

#define base xglboost model
model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

#small grid so the pipeline runs fast on GitHub Actions.
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

#model pipeline
pipeline = make_pipeline(preprocessor, model)

#start MLflow run
with mlflow.start_run():

    #hyperparameter tuning with GridSearchCV
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="recall", n_jobs=-1)
    grid.fit(Xtrain, ytrain)

    #log hyperparameters
    mlflow.log_params(grid.best_params_)

    #log all parameter combinations with their mean test scores
    results= grid.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        #log each combination
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    #log best parameters
    best_params = grid.best_params_
    mlflow.log_params(best_params)

    #store best parameter
    best_model = grid.best_estimator_
    print("Best params:", grid.best_params_)

    # Set classification threshold
    classification_threshold = 0.45   # Choose a classification threshold

    #make predictions on the training data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    #make predictions on test data
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    #evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics for best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
     })

# Save next to app.py so the Streamlit app can load it directly
m_path="tourism_project/deployment/best_tourism_model_v1.joblib"
joblib.dump(best_model,m_path)
mlflow.log_artifact(m_path,artifact_path="model")
print(f"Model saved to {m_path}")
