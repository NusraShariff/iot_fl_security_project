import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from security.crc import compute_crc
from trust_management.trust import filter_trusted_clients
from security.encryption import generate_keys, generate_shared_key, encrypt, decrypt

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
# num_clients = 7
# split_size = len(x_train_final) // num_clients
num_clients = int(input("\nEnter number of clients: "))

split_size = len(x_train_final) // num_clients

clients_data = []
for i in range(num_clients):
    start = i * split_size
    end = (i + 1) * split_size
    clients_data.append((x_train_final[start:end], y_train.iloc[start:end]))

print("\nNumber of clients:", len(clients_data))

# 5. Federated Learning
# num_rounds = 5
# global_model = SGDClassifier(loss='log_loss')
num_rounds = int(input("Enter number of rounds: "))
noise_level = float(input("Enter DP noise level: "))

global_model = SGDClassifier(loss='log_loss')

x_init, y_init = clients_data[0]
global_model.partial_fit(x_init, y_init, classes=np.array([0, 1]))

# ECC Key Setup
server_private, server_public = generate_keys()
client_keys = [generate_keys() for _ in range(num_clients)]

shared_keys = []
for priv, pub in client_keys:
    shared_keys.append(generate_shared_key(priv, server_public))

round_accuracy = []
round_loss = []

client_history = [1.0] * num_clients

for r in range(num_rounds):
    print(f"\nRound {r+1}")

    local_models = []
    accuracies = []
    crc_status = []
    history_scores = []

    for idx, (x_client, y_client) in enumerate(clients_data):
        local_model = SGDClassifier(loss='log_loss')

        local_model.partial_fit(x_client[:1], y_client.iloc[:1], classes=np.array([0, 1]))

        local_model.coef_ = global_model.coef_.copy()
        local_model.intercept_ = global_model.intercept_.copy()

        local_model.partial_fit(x_client, y_client)

        y_pred_client = local_model.predict(x_client)
        acc = accuracy_score(y_client, y_pred_client)

        # CRC check
        crc_value = compute_crc(local_model.coef_)

        if np.random.rand() < 0.1:
            print("Data corrupted ❌")
            local_model.coef_ += np.random.normal(
                0, 0.01,
                local_model.coef_.shape
            )

        new_crc = compute_crc(local_model.coef_)

        if crc_value != new_crc:
            print("CRC mismatch ❌")
            crc_passed = False
        else:
            print("CRC passed ✅")
            crc_passed = True

        # Encrypt model before sending
        key = shared_keys[idx]
        # local_model.coef_ = encrypt(local_model.coef_, key)
        encrypted_coef, iv = encrypt(local_model.coef_, key)

        local_model.encrypted_coef = encrypted_coef
        local_model.iv = iv
        local_model.shared_key = key
        local_model.original_shape = global_model.coef_.shape

        local_models.append(local_model)
        accuracies.append(acc)
        crc_status.append(crc_passed)
        history_scores.append(client_history[idx])

    # Trust filtering
    filtered_models, filtered_acc, trust_scores = filter_trusted_clients(
        local_models,
        accuracies,
        crc_status,
        history_scores
    )

    print("Trusted clients:", len(filtered_models))
    print("Trust scores:", trust_scores)

    # Update client history
    for idx in range(len(accuracies)):
        if crc_status[idx]:
            current_behavior = accuracies[idx]
        else:
            current_behavior = 0

        client_history[idx] = (
            0.7 * client_history[idx] +
            0.3 * current_behavior
        )

    print("Client history:", client_history)

    # Decrypt before aggregation
    for model in filtered_models:
        model.coef_ = decrypt(
            model.encrypted_coef,
            model.shared_key,
            model.iv,
            model.original_shape
        )

    weights = np.array(filtered_acc) / np.sum(filtered_acc)

    coef_avg = np.sum([w * m.coef_ for w, m in zip(weights, filtered_models)], axis=0)
    intercept_avg = np.sum([w * m.intercept_ for w, m in zip(weights, filtered_models)], axis=0)

    print("Client weights:", weights)

    # Differential Privacy
    noise = np.random.laplace(0, noise_level, coef_avg.shape)
    coef_avg += noise
    
    global_model.coef_ = coef_avg
    global_model.intercept_ = intercept_avg

    # Round evaluation
    y_pred_round = global_model.predict(x_test_final)
    acc_round = accuracy_score(y_test, y_pred_round)

    y_prob_round = global_model.predict_proba(x_test_final)
    loss_round = log_loss(y_test, y_prob_round)

    round_accuracy.append(acc_round)
    round_loss.append(loss_round)

# Final Evaluation
print("\nGlobal model created")

y_pred = global_model.predict(x_test_final)
accuracy = accuracy_score(y_test, y_pred)

print("\nFL Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

y_prob_global = global_model.predict_proba(x_test_final)
global_loss = log_loss(y_test, y_prob_global)

print("\nGlobal Error (Log Loss):", global_loss)

print("\nRound Accuracy:", round_accuracy)
print("Round Loss:", round_loss)