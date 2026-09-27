import math
import statistics


def percentile(values, percentile):
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (
        (len(values) - 1)
        * percentile
        / 100
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    weight = position - lower

    return (
        values[lower]
        * (1 - weight)
        + values[upper]
        * weight
    )


def calculate_latency_metrics(results):
    latencies = [
        r["latency_seconds"]
        for r in results
        if r.get("latency_seconds")
        is not None
    ]

    if not latencies:
        return {
            "mean_seconds": None,
            "median_seconds": None,
            "p95_seconds": None,
            "maximum_seconds": None,
        }

    return {
        "mean_seconds": statistics.mean(
            latencies
        ),
        "median_seconds": statistics.median(
            latencies
        ),
        "p95_seconds": percentile(
            latencies,
            95,
        ),
        "maximum_seconds": max(
            latencies
        ),
    }


def calculate_accuracy(results):
    scored = [
        r
        for r in results
        if r.get("correct") is not None
    ]

    if not scored:
        return None

    correct = sum(
        1
        for r in scored
        if r["correct"] is True
    )

    return correct / len(scored)


def calculate_capability_accuracy(results):
    grouped = {}

    for result in results:
        if result.get("correct") is None:
            continue

        capability = result[
            "capability"
        ]

        grouped.setdefault(
            capability,
            [],
        ).append(result)

    output = {}

    for capability, items in grouped.items():
        correct = sum(
            1
            for item in items
            if item["correct"] is True
        )

        output[capability] = {
            "correct": correct,
            "total": len(items),
            "accuracy": (
                correct / len(items)
            ),
        }

    return output