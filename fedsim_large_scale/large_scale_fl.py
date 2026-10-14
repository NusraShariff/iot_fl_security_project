import sys
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from unsw_data_manager import UNSWDataManager
from secure_fedavg import SecureFedAvg
from fedsim.losses import CrossEntropyLoss
from fedsim.scores import Accuracy


class UNSWModel(nn.Module):

    def __init__(self, input_size=190, num_classes=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


train_path = "../dataset/used/UNSW_NB15_training-set.csv"
test_path = "../dataset/used/UNSW_NB15_testing-set.csv"

num_clients = int(input("Enter number of clients: "))
num_rounds = int(input("Enter number of rounds: "))
dp_noise = float(input("Enter DP noise level: "))

print("Creating UNSW-NB15 data manager...")

data_manager = UNSWDataManager(
    train_path=train_path,
    test_path=test_path,
    num_clients=num_clients
)

print("Data manager created successfully!")

print("Number of clients:", num_clients)

print(
    "Global test samples:",
    len(data_manager.get_global_dataset()["test"])
)

print("\nStarting Federated Learning...")

SecureFedAvg.configured_num_clients = num_clients

algorithm = SecureFedAvg(
    data_manager=data_manager,
    metric_logger=None,
    num_clients=num_clients,
    sample_scheme="uniform",
    sample_rate=1.0,
    model_def=partial(UNSWModel, input_size=190, num_classes=2),
    epochs=1,
    criterion_def=partial(CrossEntropyLoss, log_freq=1),
    batch_size=64,
    test_batch_size=128,
    device="cpu",
    dp_noise=dp_noise,
)

algorithm.hook_global_score(
    partial(Accuracy, log_freq=1),
    score_name="accuracy",
    split_name="test"
)
algorithm.hook_local_score(
    partial(Accuracy, log_freq=1),
    score_name="accuracy",
    split_name="train"
)

results = algorithm.train(
    rounds=num_rounds
)

print("\nFederated Learning completed!")

print("\nResults:")
print(results)