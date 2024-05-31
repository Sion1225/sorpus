import pytest

from src.sorpus.sentences import find_with_word, replace_word


def test_find_with_word():
    assert find_with_word(
        "experience",
        [
            "Proverbs are short sentences drawn from long experience.",
            "Naked I came into the world, and naked I must go out.",
        ],
    ) == ["Proverbs are short sentences drawn from long experience."]


def test_replace_word():
    assert replace_word(
        "experience",
        "wisdom",
        [
            "Proverbs are short sentences drawn from long experience.",
            "Naked I came into the world, and naked I must go out.",
        ],
    ) == [
        "Proverbs are short sentences drawn from long wisdom.",
        "Naked I came into the world, and naked I must go out.",
    ]
