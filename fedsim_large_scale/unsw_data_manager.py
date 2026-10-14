from fedsim.distributed.data_management.data_manager import DataManager
from torch.utils.data import Dataset
import torch

from unsw_dataset import UNSWDataset


class CombinedUNSWDataset(Dataset):

    def __init__(self, train_dataset, test_dataset):

        self.x = torch.cat(
            [train_dataset.x, test_dataset.x],
            dim=0
        )

        self.y = torch.cat(
            [train_dataset.y, test_dataset.y],
            dim=0
        )

        self.targets = torch.cat(
            [train_dataset.y, test_dataset.y],
            dim=0
        ).numpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class UNSWDataManager(DataManager):

    def __init__(self, train_path, test_path, num_clients, seed=10):

        self.train_path = train_path
        self.test_path = test_path
        self.num_clients = num_clients

        super().__init__(
            root=".",
            seed=seed,
            save_dir=None
        )

    def get_identifiers(self):

        return [
            "UNSW_NB15",
            str(self.num_clients)
        ]

    def make_transforms(self):

        return {"train": None}, {"test": None}

    def make_datasets(self, root):

        train_dataset = UNSWDataset(
            self.train_path,
            fit=True
        )

        test_dataset = UNSWDataset(
            self.test_path,
            encoder=train_dataset.encoder,
            scaler=train_dataset.scaler,
            fit=False
        )

        self.train_size = len(train_dataset)
        self.test_size = len(test_dataset)

        combined_dataset = CombinedUNSWDataset(
            train_dataset,
            test_dataset
        )

        return combined_dataset, combined_dataset

    def partition_local_data(self, dataset):

        total_train_samples = self.train_size

        samples_per_client = total_train_samples // self.num_clients

        indices = []

        for i in range(self.num_clients):

            start = i * samples_per_client

            if i == self.num_clients - 1:
                end = total_train_samples
            else:
                end = (i + 1) * samples_per_client

            indices.append(
                list(range(start, end))
            )

        return {
            "train": indices
        }

    def partition_global_data(self, dataset):

        test_start = self.train_size
        test_end = self.train_size + self.test_size

        return {
            "test": list(range(test_start, test_end))
        }