
import pandas as pd  #import pandas for loading and manipulating tabular data
from sklearn.model_selection import train_test_split  #import utility for splitting data into training and test sets

df = pd.read_csv("tourism_project/data/tourism.csv")  #load the tourism dataset from the CSV file
df.drop(columns=["CustomerID"], inplace=True)  #remove the unique customer identifier because it is not useful for prediction

target = "ProdTaken"  #define the outcome variable to be predicted


num_features = [      #list the numeric variables used by the model
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


cat_features = [        #list the categorical variables that require encoding later
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched",
]


X = df[num_features + cat_features]  #create the feature matrix from all numerical and categorical variables


y = df[target]  #create the target vector containing the product-purchase outcome


Xtrain, Xtest, ytrain, ytest = train_test_split(  #split features and target into training and testing datasets
    X,
    y,
    test_size=0.2,  #reserve 20% of the observations for final model evaluation
    random_state=42,  #set a seed so that the split is reproducible
    stratify=y,  #preserve the original class balance of ProdTaken in both datasets
)


Xtrain.to_csv("Xtrain.csv", index=False)  #save training features without the pandas row index
Xtest.to_csv("Xtest.csv", index=False)  #save test features without the pandas row index
ytrain.to_csv("ytrain.csv", index=False)  #save training target labels without the pandas row index
ytest.to_csv("ytest.csv", index=False)  #save test target labels without the pandas row index


print("Data prepared: train/test splits written.")  #confirm that the prepared datasets were saved successfully
