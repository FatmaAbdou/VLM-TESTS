import re


def normalize(text):
    if text is None:
        return ""

    text = str(text).strip().lower()

    text = re.sub(
        r"^[\s\"'`]+|[\s\"'`]+$",
        "",
        text,
    )

    text = re.sub(
        r"[.!?,;:]+$",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def extract_number(text):
    match = re.search(
        r"\b\d+(?:\.\d+)?\b",
        str(text),
    )

    if not match:
        return None

    return match.group(0)


def score_exact(expected, predicted):
    return 1.0 if normalize(expected) == normalize(predicted) else 0.0


def score_numeric(expected, predicted):
    expected_number = extract_number(expected)
    predicted_number = extract_number(predicted)

    if expected_number is None:
        return 0.0

    if predicted_number is None:
        return 0.0

    return (
        1.0
        if expected_number == predicted_number
        else 0.0
    )


def score_choice(expected, predicted):
    expected = normalize(expected)
    predicted = normalize(predicted)

    if expected == predicted:
        return 1.0

    if expected and expected in predicted:
        return 1.0

    return 0.0


def score_spatial_partial(test, predicted):
    normalized_predicted = normalize(predicted)

    accepted_answers = [
        normalize(answer)
        for answer in test.get(
            "accepted_answers",
            [],
        )
    ]

    # Complete spatial answer.
    if normalized_predicted in accepted_answers:
        return 1.0

    # Partial spatial relations.
    partial_answers = [
        normalize(answer)
        for answer in test.get(
            "partial_answers",
            [],
        )
    ]

    if normalized_predicted in partial_answers:
        return 0.5

    return 0.0


def score_test(test, predicted):
    mode = test.get(
        "scoring_mode",
        "exact_or_semantic",
    )

    expected = test.get(
        "expected_answer"
    )

    accepted_answers = test.get(
        "accepted_answers",
        [],
    )

    if mode == "manual":
        return None

    if mode == "spatial_partial":
        return score_spatial_partial(
            test,
            predicted,
        )

    normalized_predicted = normalize(
        predicted
    )

    if accepted_answers:
        normalized_accepted = [
            normalize(answer)
            for answer in accepted_answers
        ]

        if normalized_predicted in normalized_accepted:
            return 1.0

    if mode == "choice":
        return score_choice(
            expected,
            predicted,
        )

    if test.get("answer_type") == "numeric":
        return score_numeric(
            expected,
            predicted,
        )

    return score_exact(
        expected,
        predicted,
    )