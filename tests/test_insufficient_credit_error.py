from spoon_bot.exceptions import user_friendly_error
from spoon_bot.utils.retry import is_retryable


class InsufficientCreditsError(Exception):
    pass


def test_insufficient_credit_has_top_up_message():
    error = InsufficientCreditsError("[openrouter] Insufficient credits")

    assert user_friendly_error(error) == (
        "Insufficient credits. Please top up your account and try again."
    )


def test_insufficient_credit_is_not_retryable():
    assert not is_retryable(
        InsufficientCreditsError("[openrouter] Insufficient credits")
    )
