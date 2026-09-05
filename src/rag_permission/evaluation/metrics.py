def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(top_k & relevant_set) / len(relevant_set)


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0 or not retrieved:
        return 0.0
    top_k = list(dict.fromkeys(retrieved[:k]))
    relevant_set = set(relevant)
    return len(set(top_k) & relevant_set) / min(k, len(top_k))


def reciprocal_rank(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0 or not relevant:
        return 0.0
    relevant_set = set(relevant)
    for rank, item in enumerate(retrieved[:k], 1):
        if item in relevant_set:
            return 1 / rank
    return 0.0


def mean_reciprocal_rank(
    retrieved: list[list[str]], relevant: list[list[str]], k: int
) -> float:
    if not retrieved:
        return 0.0
    values = [
        reciprocal_rank(items, targets, k)
        for items, targets in zip(retrieved, relevant, strict=True)
    ]
    return sum(values) / len(values)
