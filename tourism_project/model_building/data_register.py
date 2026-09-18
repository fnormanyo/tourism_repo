
#import pandas for loading tabular data
import pandas as pd

#define the file path for the raw tourism dataset
RAW_PATH = "tourism_project/data/tourism.csv"

#load the raw dataset into a pandas DataFrame
df = pd.read_csv(RAW_PATH)


#list the columns required for the tourism machine-learning project
data_columns = [
    'CustomerID',                  # Unique identifier assigned to each customer
    'ProdTaken',                   # Target variable indicating whether the customer purchased the product
    'Age',                         # Customer's age in years
    'TypeofContact',               # Method used to contact the customer
    'CityTier',                    # Tier classification of the customer's city
    'Occupation',                  # Customer's employment or occupation category
    'Gender',                      # Customer's gender
    'NumberOfPersonVisiting',      # Number of people included in the planned trip
    'PreferredPropertyStar',       # Customer's preferred hotel/property star rating
    'MaritalStatus',               # Customer's marital-status category
    'NumberOfTrips',               # Number of trips previously taken by the customer
    'Passport',                    # Indicator showing whether the customer owns a passport
    'OwnCar',                      # Indicator showing whether the customer owns a car
    'NumberOfChildrenVisiting',    # Number of children expected to join the trip
    'Designation',                 # Customer's job designation or employment level
    'MonthlyIncome',               # Customer's monthly income
    'PitchSatisfactionScore',      # Customer satisfaction score for the sales pitch
    'ProductPitched',              # Tourism product offered to the customer
    'NumberOfFollowups',           # Number of follow-up contacts made with the customer
    'DurationOfPitch'              # Length of the sales pitch delivered to the customer
]


#identify expected columns that are absent from the loaded dataset
missing = [c for c in data_columns if c not in df.columns]


#stop execution if the dataset does not contain all required columns
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")


#confirm that the dataset passed the column-validation check
print("Dataset registered successfully.")

#display the number of rows and columns in the dataset
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

#display all available column names for verification
print("Columns:", list(df.columns))

#display the class distribution of tourism product purchases
print("Purchase distribution:")

#count customers who did and did not purchase the tourism product
print(df["ProdTaken"].value_counts())
