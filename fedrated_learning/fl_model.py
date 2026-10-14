import pandas as pd
import numpy as np
import zlib

def compute_crc(data):
    return zlib.crc32(data.tobytes())

from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
train_data = pd.read_csv("../dataset/used/UNSW_NB15_training-set.csv")
test_data = pd.read_csv("../dataset/used/UNSW_NB15_testing-set.csv")

# 2. Separate Features and Labels
drop_cols = ["label", "attack_cat"]
if "id" in train_data.columns:
    drop_cols.append("id")

x_train = train_data.drop(columns=drop_cols)
y_train = train_data["label"]
x_test = test_data.drop(columns=drop_cols)
y_test = test_data["label"]

# 3. Preprocessing
categorical_cols = ["proto", "service", "state"]
numeric_cols = x_train.columns.difference(categorical_cols)

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
x_train_cat = encoder.fit_transform(x_train[categorical_cols])
x_test_cat = encoder.transform(x_test[categorical_cols])

scaler = StandardScaler()
x_train_num = scaler.fit_transform(x_train[numeric_cols])
x_test_num = scaler.transform(x_test[numeric_cols])

x_train_final = np.hstack([x_train_num, x_train_cat])
x_test_final = np.hstack([x_test_num, x_test_cat])

print("train shape:", x_train_final.shape)
print("test shape:", x_test_final.shape)

# Shuffle data
indices = np.random.permutation(len(x_train_final))
x_train_final = x_train_final[indices]
y_train = y_train.iloc[indices]

# 4. Split into clients
num_clients = 3
split_size = len(x_train_final) // num_clients

clients_data = []
for i in range(num_clients):
    start = i * split_size
    end = (i + 1) * split_size
    clients_data.append((x_train_final[start:end], y_train.iloc[start:end]))

print("\nNumber of clients:", len(clients_data))

# 5. Federated Learning
num_rounds = 5
global_model = SGDClassifier(loss='log_loss')

# initialize
x_init, y_init = clients_data[0]
global_model.partial_fit(x_init, y_init, classes=np.array([0, 1]))

# 🔥 NEW: tracking for graphs
round_accuracy = []
round_loss = []

for r in range(num_rounds):
    print(f"\nRound {r+1}")

    local_models = []
    accuracies = []

    # ---- Local Training ----
    for x_client, y_client in clients_data:
        local_model = SGDClassifier(loss='log_loss')

        # initialize
        local_model.partial_fit(x_client[:1], y_client.iloc[:1], classes=np.array([0, 1]))

        # copy global weights
        local_model.coef_ = global_model.coef_.copy()
        local_model.intercept_ = global_model.intercept_.copy()

        # train
        local_model.partial_fit(x_client, y_client)

        # compute accuracy
        y_pred_client = local_model.predict(x_client)
        acc = accuracy_score(y_client, y_pred_client)

        # ---- CRC CHECK ----
        crc_value = compute_crc(local_model.coef_)

        if np.random.rand() < 0.1:
            print("Data corrupted ❌")
            local_model.coef_ += np.random.normal(0, 0.01, local_model.coef_.shape)

        new_crc = compute_crc(local_model.coef_)

        if crc_value != new_crc:
            print("CRC mismatch — rejecting client ❌")
            continue

        local_models.append(local_model)
        accuracies.append(acc)

    # ---- Trust Filtering ----
    trust_threshold = 0.6
    filtered_models = []
    filtered_acc = []

    for model, acc in zip(local_models, accuracies):
        if acc >= trust_threshold:
            filtered_models.append(model)
            filtered_acc.append(acc)

    print("Trusted clients:", len(filtered_models))

    if len(filtered_models) == 0:
        filtered_models = local_models
        filtered_acc = accuracies

    # ---- Weighted Aggregation ----
    weights = np.array(filtered_acc) / np.sum(filtered_acc)

    coef_avg = np.sum([w * m.coef_ for w, m in zip(weights, filtered_models)], axis=0)
    intercept_avg = np.sum([w * m.intercept_ for w, m in zip(weights, filtered_models)], axis=0)

    print("Client weights:", weights)

    # ---- Differential Privacy ----
    noise = np.random.laplace(0, 0.01, coef_avg.shape)
    coef_avg += noise

    # update global model
    global_model.coef_ = coef_avg
    global_model.intercept_ = intercept_avg

    # 🔥 NEW: Round-wise evaluation
    y_pred_round = global_model.predict(x_test_final)
    acc_round = accuracy_score(y_test, y_pred_round)

    y_prob_round = global_model.predict_proba(x_test_final)
    loss_round = log_loss(y_test, y_prob_round)

    round_accuracy.append(acc_round)
    round_loss.append(loss_round)

# 6. Evaluation
print("\nGlobal model created")

y_pred = global_model.predict(x_test_final)
accuracy = accuracy_score(y_test, y_pred)

print("\nFL Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Global loss
y_prob_global = global_model.predict_proba(x_test_final)
global_loss = log_loss(y_test, y_prob_global)

print("\nGlobal Error (Log Loss):", global_loss)

# NEW: Print round values (for MATLAB)
print("\nRound Accuracy:", round_accuracy)
print("Round Loss:", round_loss)

