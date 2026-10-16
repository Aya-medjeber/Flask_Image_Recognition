"""Tests for the preprocessing and prediction helpers in model.py."""
import numpy as np
import pytest
from PIL import Image

import model
from model import preprocess_img, predict_result

SAMPLE_IMAGE = "test_images/3/Sign 3 (30).jpeg"


def save_image(folder, mode, size=(100, 80)):
    """Create an image file of the given colour mode and return its path."""
    path = folder / f"image_{mode}.png"
    Image.new(mode, size).save(path)
    return path


def test_preprocess_output_shape():
    """A real photo is turned into one 224x224 RGB image."""
    assert preprocess_img(SAMPLE_IMAGE).shape == (1, 224, 224, 3)


def test_preprocess_values_are_normalised():
    """Pixel values are scaled from 0-255 down to 0-1."""
    result = preprocess_img(SAMPLE_IMAGE)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_resizes_any_size(tmp_path):
    """Edge case: a small, non-square image is resized to 224x224."""
    path = save_image(tmp_path, "RGB", size=(30, 50))
    assert preprocess_img(path).shape == (1, 224, 224, 3)


def test_preprocess_grayscale_image(tmp_path):
    """Edge case: a 1-channel grayscale image still gives 3 channels."""
    assert preprocess_img(save_image(tmp_path, "L")).shape == (1, 224, 224, 3)


def test_preprocess_rgba_image(tmp_path):
    """Edge case: a PNG with transparency (4 channels) still gives 3 channels."""
    assert preprocess_img(save_image(tmp_path, "RGBA")).shape == (1, 224, 224, 3)


def test_preprocess_missing_file_raises():
    """A path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("does/not/exist.png")


class FakeModel:  # pylint: disable=too-few-public-methods
    """Stand-in for the Keras model so the test runs without TensorFlow."""

    def predict(self, _images):
        """Return fake probabilities where digit 4 is the most likely."""
        probabilities = np.zeros((1, 10))
        probabilities[0][4] = 0.9
        return probabilities


def test_predict_result_returns_most_likely_digit(monkeypatch):
    """predict_result returns the index of the highest probability (mocked model)."""
    monkeypatch.setattr(model, "model", FakeModel())
    assert predict_result(np.zeros((1, 224, 224, 3))) == 4


def test_real_model_predicts_sample():
    """The real model predicts 3 for a real 'Sign 3' photo."""
    prediction = predict_result(preprocess_img(SAMPLE_IMAGE))
    assert isinstance(prediction, np.integer)
    assert int(prediction) == 3
