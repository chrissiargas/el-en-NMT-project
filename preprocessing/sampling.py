from typing import Dict, Optional


def allocate(available: Dict[str, int], total: Optional[int], temperature: float = 1.0) -> Dict[str, int]:
    """Split `total` rows across corpora with probabilities proportional to size ** (1 / temperature).

    A corpus never gets more than it has; its unused share goes to the others (water-filling).
    temperature = 1 is proportional to size, larger values flatten towards uniform.
    """
    if not total:
        return dict(available)

    quotas = {}
    remaining = {k: v for k, v in available.items() if v > 0}
    budget = total
    while remaining and budget > 0:
        weights = {k: v ** (1 / temperature) for k, v in remaining.items()}
        norm = sum(weights.values())
        share = {k: budget * w / norm for k, w in weights.items()}
        full = [k for k in remaining if share[k] >= remaining[k]]
        if not full:
            quotas.update({k: int(share[k]) for k in remaining})
            break
        for k in full:
            quotas[k] = remaining.pop(k)
            budget -= quotas[k]

    return {k: quotas.get(k, 0) for k in available}


if __name__ == '__main__':
    print(allocate({'HPLT': 30_000_000, 'CCMatrix': 25_000_000, 'ParaCrawl': 9_000_000, 'MaCoCu': 1_500_000},
                   total=20_000_000, temperature=1.2))
