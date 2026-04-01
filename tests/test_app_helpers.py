from app_helpers import (
    QUALITY_CHOICES,
    FORMAT_CHOICES,
    LANGUAGE_CHOICES,
    quality_to_model,
    language_choice_to_code,
)


def test_quality_choices_has_three_options():
    assert len(QUALITY_CHOICES) == 3


def test_quality_to_model_fast():
    assert quality_to_model("Fast (lower quality)") == "tiny"


def test_quality_to_model_balanced():
    assert quality_to_model("Balanced (recommended)") == "small"


def test_quality_to_model_best():
    assert quality_to_model("Best (slower)") == "large-v3"


def test_quality_to_model_default_on_unknown():
    assert quality_to_model("garbage") == "small"


def test_format_choices_contains_all_formats():
    labels = [label for label, _ in FORMAT_CHOICES]
    assert "TXT (plain text)" in labels
    assert "SRT (subtitles)" in labels
    assert "VTT (web subtitles)" in labels
    assert "JSON (structured)" in labels


def test_language_auto_detect_maps_to_none():
    assert language_choice_to_code("Auto-detect") is None


def test_language_english_maps_to_code():
    assert language_choice_to_code("en — English") == "en"


def test_language_spanish_maps_to_code():
    assert language_choice_to_code("es — Spanish") == "es"
