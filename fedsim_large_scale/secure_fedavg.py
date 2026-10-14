import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from fedsim.distributed.centralized.training.fedavg import FedAvg
from security.crc import compute_crc
from trust_management.trust import calculate_trust_score
from security.encryption import (
    generate_keys,
    generate_shared_key,
    encrypt,
    decrypt
)


class SecureFedAvg(FedAvg):

    @staticmethod
    def init(server_storage, dp_noise=0.01, *args, **kwargs):
        SecureFedAvg.dp_noise = dp_noise
        FedAvg.init(server_storage)

        server_private, server_public = generate_keys()

        num_clients = SecureFedAvg.configured_num_clients

        client_keys = [
            generate_keys()
            for _ in range(num_clients)
        ]

        shared_keys = []

        for private_key, public_key in client_keys:
            shared_key = generate_shared_key(
                private_key,
                server_public
            )
            shared_keys.append(shared_key)

        server_storage.write(
            "shared_keys",
            shared_keys
        )

        server_storage.write(
            "client_history",
            [1.0] * len(shared_keys)
        )

        server_storage.write(
             "accepted_updates",
            []
        )


    @staticmethod
    def send_to_client(server_storage, client_id):

        context = FedAvg.send_to_client(
            server_storage,
            client_id
        )

        shared_keys = server_storage.read("shared_keys")

        shared_keys = server_storage.read("shared_keys")

        print("Client ID:", client_id)
        print("Number of shared keys:", len(shared_keys))

        context["shared_key"] = shared_keys[client_id]

        return context

    @staticmethod
    def send_to_server(
        id,
        rounds,
        storage,
        datasets,
        train_split_name,
        scores,
        epochs,
        criterion,
        train_batch_size,
        inference_batch_size,
        optimizer_def,
        lr_scheduler_def=None,
        device="cuda",
        ctx=None,
        step_closure=None
    ):

        result = FedAvg.send_to_server(
            id,
            rounds,
            storage,
            datasets,
            train_split_name,
            scores,
            epochs,
            criterion,
            train_batch_size,
            inference_batch_size,
            optimizer_def,
            lr_scheduler_def,
            device,
            ctx,
            step_closure
        )

        local_params = result["local_params"]

        key = ctx["shared_key"]

        encrypted_params, iv = encrypt(
            local_params.detach().cpu().numpy(),
            key
        )

        result["encrypted_params"] = encrypted_params
        result["iv"] = iv
        result["param_shape"] = tuple(local_params.shape)

        del result["local_params"]

        print(f"Client {id}: Model encrypted 🔐")

        return result

    @staticmethod
    def receive_from_client(
        server_storage,
        client_id,
        client_msg,
        train_split_name,
        serial_aggregator,
        appendix_aggregator
    ):

        encrypted_params = client_msg["encrypted_params"]
        iv = client_msg["iv"]
        param_shape = client_msg["param_shape"]

        shared_keys = server_storage.read("shared_keys")
        key = shared_keys[client_id]

        decrypted_params = decrypt(
            encrypted_params,
            key,
            iv,
            param_shape,
            dtype=np.float32
        )

        local_params = torch.tensor(
            decrypted_params,
            dtype=torch.float32
        )

        original_crc = compute_crc(
            local_params.detach().cpu().numpy()
        )

        if np.random.rand() < 0.1:

            print(f"Client {client_id}: Data corrupted ❌")

            local_params += torch.normal(
                mean=0.0,
                std=0.01,
                size=local_params.shape
            )

        new_crc = compute_crc(
            local_params.detach().cpu().numpy()
        )

        if original_crc != new_crc:
            print(f"Client {client_id}: CRC mismatch ❌")
            print(f"Client {client_id}: Update rejected")

            return True

        print(f"Client {client_id}: CRC passed ✅")

        client_history = server_storage.read("client_history")

        history_score = client_history[client_id]

        accuracy = client_msg["metrics"]["train"]["accuracy"]

        trust_score = calculate_trust_score(
            accuracy,
            True,
            history_score
        )

        print(
            f"Client {client_id}: "
            f"Accuracy={accuracy:.3f}, "
            f"History={history_score:.3f}, "
            f"Trust={trust_score:.3f}"
        )

        if trust_score >= 0.6:
            print(f"Client {client_id}: Trusted ✅")
        else:
            print(f"Client {client_id}: Not trusted ❌")

        client_history[client_id] = (
            0.7 * client_history[client_id] +
            0.3 * accuracy
        )

        server_storage.write(
            "client_history",
            client_history
        )

        if trust_score < 0.6:
            print(f"Client {client_id}: Update rejected")

            return True


        accepted_updates = server_storage.read(
            "accepted_updates"
        )

        accepted_updates.append(
            (
                local_params.detach().clone(),
                accuracy
            )
        )

        server_storage.write(
            "accepted_updates",
            accepted_updates
        )

        return True

    @staticmethod
    def optimize(server_storage, *args, **kwargs):

        accepted_updates = server_storage.read(
            "accepted_updates"
        )

        if len(accepted_updates) == 0:
            print("No trusted updates to aggregate.")
            return

        accuracies = np.array(
            [accuracy for _, accuracy in accepted_updates]
        )

        weights = accuracies / np.sum(accuracies)

        print("Client aggregation weights:", weights)

        weighted_update = sum(
            weight * update
            for weight, (update, _) in zip(
                weights,
                accepted_updates
            )
        )

        cloud_params = server_storage.read(
            "cloud_params"
        )

        with torch.no_grad():
            cloud_params.copy_(weighted_update)

        dp_noise = SecureFedAvg.dp_noise

        noise = torch.tensor(
            np.random.laplace(
                0,
                dp_noise,
                size=cloud_params.shape
            ),
            dtype=cloud_params.dtype
        )

        with torch.no_grad():
            cloud_params.add_(noise)

        print(
            f"DP noise added: {dp_noise}"
        )

        server_storage.write(
            "accepted_updates",
            []
        )

        return {}