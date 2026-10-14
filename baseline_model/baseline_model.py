# import pandas as pd

# # load training and testing datasets
# train_data = pd.read_csv("../dataset/used/UNSW_NB15_training-set.csv")
# test_data = pd.read_csv("../dataset/used/UNSW_NB15_testing-set.csv")

# # display basic information
# print("training data shape:", train_data.shape)
# print("testing data shape:", test_data.shape)

# print("\ntraining data columns:")
# print(train_data.columns)

# # check label distribution
# print("\nlabel distribution in training data:")
# print(train_data["label"].value_counts())

# print("\nlabel distribution in testing data:")
# print(test_data["label"].value_counts())

# # separate features and labels
# x_train = train_data.drop(columns=["label", "attack_cat"])
# y_train = train_data["label"]

# x_test = test_data.drop(columns=["label", "attack_cat"])
# y_test = test_data["label"]

# print("\nfeature set shape:", x_train.shape)
# print("label set shape:", y_train.shape)

# from sklearn.preprocessing import OneHotEncoder

# # identify categorical columns
# categorical_cols = ["proto", "service", "state"]

# # one-hot encoding
# encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# # fit on training data and transform both train and test
# x_train_encoded = encoder.fit_transform(x_train[categorical_cols])
# x_test_encoded = encoder.transform(x_test[categorical_cols])

# # drop categorical columns from original data
# x_train_numeric = x_train.drop(columns=categorical_cols)
# x_test_numeric = x_test.drop(columns=categorical_cols)

# # combine numeric and encoded categorical features
# import numpy as np
# x_train_final = np.hstack([x_train_numeric.values, x_train_encoded])
# x_test_final = np.hstack([x_test_numeric.values, x_test_encoded])

# print("\nfinal training feature shape:", x_train_final.shape)
# print("final testing feature shape:", x_test_final.shape)

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

# # initialize model
# model = LogisticRegression(max_iter=1000)

# # train model
# model.fit(x_train_final, y_train)

# # make predictions
# y_pred = model.predict(x_test_final)

# # evaluate model
# accuracy = accuracy_score(y_test, y_pred)
# print("\nbaseline model accuracy:", accuracy)

# print("\nclassification report:")
# print(classification_report(y_test, y_pred))

# # save baseline results
# with open("../evaluation/baseline_results.txt", "w") as f:
#     f.write(f"Baseline Accuracy: {accuracy}\n")
#     f.write(classification_report(y_test, y_pred))


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
train_data = pd.read_csv("../dataset/used/UNSW_NB15_training-set.csv")
test_data = pd.read_csv("../dataset/used/UNSW_NB15_testing-set.csv")

# 2. Separate Features and Labels
# Dropping 'id' if it exists, along with 'label' and 'attack_cat'
drop_cols = ["label", "attack_cat"]
if "id" in train_data.columns: drop_cols.append("id")

x_train = train_data.drop(columns=drop_cols)
y_train = train_data["label"]
x_test = test_data.drop(columns=drop_cols)
y_test = test_data["label"]

# 3. Preprocessing: Categorical + Scaling
categorical_cols = ["proto", "service", "state"]
numeric_cols = x_train.columns.difference(categorical_cols)

# One-Hot Encoding
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
x_train_cat = encoder.fit_transform(x_train[categorical_cols])
x_test_cat = encoder.transform(x_test[categorical_cols])

# Scaling Numeric Data (Crucial for Logistic Regression!)
scaler = StandardScaler()
x_train_num = scaler.fit_transform(x_train[numeric_cols])
x_test_num = scaler.transform(x_test[numeric_cols])

# Combine
x_train_final = np.hstack([x_train_num, x_train_cat])
x_test_final = np.hstack([x_test_num, x_test_cat])

# 4. Train Model
# Increased max_iter to ensure it finishes finding the best weights
model = LogisticRegression(max_iter=2000)
model.fit(x_train_final, y_train)

# 5. Evaluate
y_pred = model.predict(x_test_final)
accuracy = accuracy_score(y_test, y_pred)

print(f"Improved Baseline Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save to your evaluation folder
with open("../evaluation/baseline_results.txt", "w") as f:
    f.write(f"Baseline Accuracy: {accuracy}\n")
    f.write(classification_report(y_test, y_pred))