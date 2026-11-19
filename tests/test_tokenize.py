import inspect

from findex.tokenize import tokenize


def test_is_generator():
    assert inspect.isgeneratorfunction(tokenize)


def test_empty_and_whitespace():
    assert list(tokenize("")) == []
    assert list(tokenize("   \n\t  ")) == []


def test_mixed_case_and_casefold():
    assert list(tokenize("Python SEARCH Straße")) == ["python", "search", "strasse"]


def test_cyrillic_and_ukrainian_apostrophe():
    text = "Київ — столиця України. П'ять козаків і сім’я."
    assert list(tokenize(text)) == ["київ", "столиця", "україни", "п'ять", "козаків", "і", "сім’я"]


def test_unicode_nfc_normalization():
    composed = list(tokenize("café"))
    decomposed = list(tokenize("cafe\u0301"))
    assert composed == decomposed == ["café"]


def test_apostrophes_and_hyphens():
    text = "Don't stop state-of-the-art 'quoted' -dash-"
    assert list(tokenize(text)) == ["don't", "stop", "state-of-the-art", "quoted", "dash"]


def test_numbers_and_alphanumeric():
    assert list(tokenize("Lab 01 in 2026 uses UTF-8 and mp3.")) == [
        "lab",
        "01",
        "in",
        "2026",
        "uses",
        "utf-8",
        "and",
        "mp3",
    ]


def test_punctuation_stripping():
    assert list(tokenize("Hello, world! (Testing... [100%])")) == [
        "hello",
        "world",
        "testing",
        "100",
    ]