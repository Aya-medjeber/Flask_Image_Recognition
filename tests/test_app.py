"""Tests for the Flask routes defined in app.py."""
import io

from PIL import Image

SAMPLE_IMAGE = "test_images/3/Sign 3 (30).jpeg"
ERROR_TEXT = b"File cannot be processed."


def make_png(mode="RGB"):
    """Return an in-memory PNG image that can be sent as an upload."""
    buffer = io.BytesIO()
    Image.new(mode, (64, 64), "white").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def post_file(client, file_obj, filename):
    """Send a file to /prediction the same way the HTML form does."""
    return client.post(
        "/prediction",
        data={"file": (file_obj, filename)},
        content_type="multipart/form-data",
    )


# ---------- Home page ----------

def test_home_page_loads(client):
    """GET / returns 200 and shows the app title."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hand Sign Digit Language Detection" in response.data


def test_home_page_has_upload_form(client):
    """The home page contains a file input that posts to /prediction."""
    response = client.get("/")
    assert b'name="file"' in response.data
    assert b'action="/prediction"' in response.data


# ---------- Valid requests ----------

def test_valid_image_shows_prediction(client, monkeypatch):
    """A valid image shows the predicted digit (model is mocked)."""
    monkeypatch.setattr("app.predict_result", lambda _img: 7)
    response = post_file(client, make_png(), "hand.png")
    assert response.status_code == 200
    assert b'font-weight-bold">7</h2>' in response.data
    assert ERROR_TEXT not in response.data


def test_real_sample_image_end_to_end(client):
    """A real 'Sign 3' photo goes through the real model and predicts 3."""
    with open(SAMPLE_IMAGE, "rb") as image_file:
        response = post_file(client, image_file, "sign3.jpeg")
    assert response.status_code == 200
    assert b'font-weight-bold">3</h2>' in response.data


def test_grayscale_png_is_accepted(client, monkeypatch):
    """Edge case: a black-and-white PNG is still a valid image."""
    monkeypatch.setattr("app.predict_result", lambda _img: 5)
    response = post_file(client, make_png("L"), "gray.png")
    assert b'font-weight-bold">5</h2>' in response.data


# ---------- Missing / invalid input ----------

def test_missing_file_shows_error(client):
    """Submitting the form with no file shows the error message."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert ERROR_TEXT in response.data


def test_empty_upload_shows_error(client):
    """An empty file (no file chosen in the browser) shows the error message."""
    response = post_file(client, io.BytesIO(b""), "")
    assert ERROR_TEXT in response.data


def test_text_file_shows_error(client):
    """A non-image file is rejected with the error message, not a crash."""
    response = post_file(client, io.BytesIO(b"this is not an image"), "notes.txt")
    assert response.status_code == 200
    assert ERROR_TEXT in response.data


def test_get_on_prediction_not_allowed(client):
    """/prediction only accepts POST, so GET returns 405."""
    response = client.get("/prediction")
    assert response.status_code == 405
