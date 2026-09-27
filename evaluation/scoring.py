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
    return (
        normalize(expected)
        == normalize(predicted)
    )


def score_numeric(expected, predicted):
    expected_number = extract_number(
        expected
    )

    predicted_number = extract_number(
        predicted
    )

    if (
        expected_number is None
        or predicted_number is None
    ):
        return False

    return (
        expected_number
        == predicted_number
    )


def score_choice(expected, predicted):
    expected = normalize(expected)
    predicted = normalize(predicted)

    if expected == predicted:
        return True

    # Handle model responses such as:
    # "The answer is sphere."
    if expected and expected in predicted:
        return True

    return False


def score_test(test, predicted):
    mode = test.get(
        "scoring_mode",
        "exact_or_semantic",
    )

    expected = test.get(
        "expected_answer"
    )

    if mode == "manual":
        return None

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