import pytest
import os

from src.sorpus.tokenize_n_lemmatize import TokenizerNLemmatizer


def test_text_lemmatizer():
    source_directory = os.path.dirname(os.path.abspath(__file__))
    stanford_path = os.path.join(source_directory, "stanford")

    tomatizer = TokenizerNLemmatizer(
        model_path=os.path.join(stanford_path, "english.muc.7class.distsim.crf.ser.gz"),
        jar_path=os.path.join(stanford_path, "stanford-ner-4.2.0.jar"),
    )

    assert tomatizer.text_lemmatizer(
        "In New•York, I saw police•officers and firefighters.",
        space_token="•",
        return_pos=False,
    ) == [
        "in",
        "New York",
        ",",
        "I",
        "see",
        "police officer",
        "and",
        "firefighter",
        ".",
    ]
    assert tomatizer.text_lemmatizer(
        "In New•York, I saw police•officers and firefighters.",
        space_token="•",
        return_pos=True,
    ) == [
        ("in", "IN"),
        ("New York", "NNP"),
        (",", ","),
        ("I", "PRP"),
        ("see", "VBP"),
        ("police officer", "NN"),
        ("and", "CC"),
        ("firefighter", "NN"),
        (".", "."),
    ]
    assert tomatizer.text_lemmatizer(
        "In New•York, I saw police•officers and firefighters."
    ) == [
        "in",
        "New York",
        ",",
        "I",
        "see",
        "police officer",
        "and",
        "firefighter",
        ".",
    ]


# def test_texts_lemmatizer():
