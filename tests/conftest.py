"""Shared pytest fixtures for the hand sign digit Flask app."""
import pytest

from app import app as flask_app


@pytest.fixture(name="client")
def client_fixture():
    """Provide a Flask test client so tests can send fake HTTP requests."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as test_client:
        yield test_client
