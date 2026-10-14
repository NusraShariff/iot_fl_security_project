# def filter_trusted_clients(local_models, accuracies, threshold=0.6):
#     filtered_models = []
#     filtered_acc = []

#     for model, acc in zip(local_models, accuracies):
#         if acc >= threshold:
#             filtered_models.append(model)
#             filtered_acc.append(acc)

#     if len(filtered_models) == 0:
#         return local_models, accuracies

#     return filtered_models, filtered_acc

def calculate_trust_score(accuracy, crc_passed, history_score):
    accuracy_score = accuracy
    crc_score = 1 if crc_passed else 0

    trust_score = (
        0.5 * accuracy_score +
        0.3 * crc_score +
        0.2 * history_score
    )

    return trust_score


def filter_trusted_clients(
    local_models,
    accuracies,
    crc_status,
    history_scores,
    threshold=0.6
):
    filtered_models = []
    filtered_acc = []
    trust_scores = []

    for model, acc, crc_passed, history_score in zip(
        local_models,
        accuracies,
        crc_status,
        history_scores
    ):
        trust_score = calculate_trust_score(
            acc,
            crc_passed,
            history_score
        )

        print(f"Client trust score: {trust_score:.3f}")

        if trust_score >= threshold:
            filtered_models.append(model)
            filtered_acc.append(acc)
            trust_scores.append(trust_score)

    if len(filtered_models) == 0:
        return local_models, accuracies, trust_scores

    return filtered_models, filtered_acc, trust_scores