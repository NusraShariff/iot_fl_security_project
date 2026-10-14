import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class UNSWDataset(Dataset):
    def __init__(self, csv_path, encoder=None, scaler=None, fit=False):
        data = pd.read_csv(csv_path)

        drop_cols = ["label", "attack_cat"]

        if "id" in data.columns:
            drop_cols.append("id")

        x = data.drop(columns=drop_cols)
        y = data["label"].values

        categorical_cols = ["proto", "service", "state"]
        numeric_cols = x.columns.difference(categorical_cols)

        if fit:
            self.encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
            self.scaler = StandardScaler()

            x_cat = self.encoder.fit_transform(x[categorical_cols])
            x_num = self.scaler.fit_transform(x[numeric_cols])
        else:
            self.encoder = encoder
            self.scaler = scaler

            x_cat = self.encoder.transform(x[categorical_cols])
            x_num = self.scaler.transform(x[numeric_cols])

        x_final = np.hstack([x_num, x_cat])

        self.x = torch.tensor(x_final, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        self.targets = y
        self.num_features = x_final.shape[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]