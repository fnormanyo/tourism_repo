import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

unnamed_columns = [
    column for column in df.columns
    if str(column).strip().lower().startswith("unnamed:")
]

df = df.drop(columns=unnamed_columns, errors="ignore")

# Validate that the expected columns are present before registering it
data_columns = [
    'CustomerID',
    'ProdTaken',
    'Age',
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'MaritalStatus',
    'NumberOfTrips',
    'Passport',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'Designation',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'ProductPitched',
    'NumberOfFollowups',
    'DurationOfPitch'
]
missing = [c for c in data_columns if c not in df.columns]

if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Purchase distribution:")
print(df["ProdTaken"].value_counts())
