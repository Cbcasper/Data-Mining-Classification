import os
from loguru import logger

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

HIGH_INCOME_PROFIT = 970
HIGH_INCOME_ACCEPTANCE_RATE = 0.1
LOW_INCOME_PROFIT = 320
LOW_INCOME_ACCEPTANCE_RATE = 0.05
EXISTING_CUSTOMERS = "existing-customers.xlsx"
POTENTIAL_CUSTOMERS = "potential-customers.xlsx"

def display(title, message):
    print()
    print(title, message)

def load_data(file):
    dataframe = pd.read_excel(os.path.join("data", file), index_col=0)
    dataframe = dataframe.drop("education", axis=1)

    number_of_rows, _ = dataframe.shape
    missing_values = str(dataframe.isna().sum(axis=0)).replace("\n", "\n\t\t")
    isna_counts = dataframe.isna().sum(axis=1).value_counts()

    print()
    print("-" * 25)
    print(f"Loading dataset {file}")
    print(f"\tNumber of rows: {number_of_rows}")
    print(f"\tNumber of missing values per feature:\n\t\t{missing_values}")
    print(f"\tNumber of rows with a missing value: {number_of_rows - isna_counts.iloc[0]}")
    print(f"\t{number_of_rows - isna_counts.iloc[0]}/{number_of_rows} or " + \
          f"{(number_of_rows - isna_counts.iloc[0]) / number_of_rows * 100:.2f} percent rows with missing values")
    return dataframe

def run_classifier(classifier, features, labels):
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)
    classifier.fit(train_features, train_labels)
    return classifier, classifier.score(test_features, test_labels)

def encode(dataframe):
    categorical_columns = ["workclass", "marital-status", "occupation",
                           "relationship", "race", "sex", "native-country", "class"]

    encoded_dataframe = np.zeros(dataframe.shape)
    for index, column_name in enumerate(dataframe.columns):
        column = dataframe.loc[:, column_name].to_numpy()
        if column_name in categorical_columns:
            encoder = OrdinalEncoder()
            column = encoder.fit_transform(column.reshape(-1, 1)).reshape(1, -1)
        encoded_dataframe[:, index] = column
    return encoded_dataframe, encoder

def get_imputer(dataframe):
    imputer = KNNImputer()
    imputer.fit(dataframe[:, :-1])
    return imputer

def write_customers(filename, predictions):
    with open(filename, "w") as file:
        for index, prediction in enumerate(predictions):
            if prediction == ">50K":
                file.write(f"Row{index}\n")

def compute_profit(predictions, accuracy):
    customers = np.count_nonzero(predictions == ">50K")
    high_income = HIGH_INCOME_PROFIT * HIGH_INCOME_ACCEPTANCE_RATE
    low_income = LOW_INCOME_PROFIT * LOW_INCOME_ACCEPTANCE_RATE
    total_expected_profit = high_income * accuracy * customers - low_income * (1 - accuracy) * customers
    print("-" * 25)
    print(f"Results")
    print(f"\tClassifier test accuracy: {accuracy:.2f}")
    print(f"\tNumbers of potential customers to be recruited: {customers}")
    print(f"\tTotal expected profit: {total_expected_profit:.2f}")

@logger.catch
def main():
    existing_customers = load_data(EXISTING_CUSTOMERS)

    encoded_dataframe, class_encoder = encode(existing_customers)
    imputer = get_imputer(encoded_dataframe)

    features = imputer.transform(encoded_dataframe[:, :-1])
    classifier, accuracy = run_classifier(RandomForestClassifier(), features, encoded_dataframe[:, -1])

    potential_customers = load_data(POTENTIAL_CUSTOMERS)
    predictions = classifier.predict(imputer.transform(encode(potential_customers)[0]))
    predictions = class_encoder.inverse_transform(predictions.reshape(-1, 1))
    write_customers(os.path.join("results", "customers.txt"), predictions)
    compute_profit(predictions, accuracy)

if __name__ == "__main__":
    main()