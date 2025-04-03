import os
import io
import pytest
from fastapi.testclient import TestClient
from app.main import app

TEST_IMAGE_FILE_PATH = os.path.join("tests", "test_image.png")

@pytest.fixture(scope="module")
def test_client():
    with TestClient(app) as client:
        yield client

@pytest.fixture
def trial_image_data():
    """
    Returns the bytes of a trial image file.
    If the file does not exist, the test is skipped.
    """
    if not os.path.exists(TEST_IMAGE_FILE_PATH):
        pytest.skip("Trial image file not found.")
    with open(TEST_IMAGE_FILE_PATH, "rb") as file:
        return file.read()

def test_predict_with_trial_image(trial_image_data, test_client):
    """
    Test the /predict endpoint using a trial image file.
    """
    files = {"file": ("trial.png", trial_image_data, "image/png")}
    response = test_client.post("/predict", files=files)
    assert response.status_code == 200, "Expected 200 OK for a valid image input."
    json_response = response.json()
    assert "prediction" in json_response, "Response JSON must contain a 'prediction' key."
    print("Prediction with trial image:", json_response["prediction"])

def test_predict_no_file(test_client):
    """
    Test the /predict endpoint when no file is provided.
    """
    response = test_client.post("/predict", data={})
    assert response.status_code == 422, "Expected 422 when required file is missing."

def test_predict_invalid_file(test_client):
    """
    Test the /predict endpoint with an invalid non-image file.
    """
    invalid_file_content = io.BytesIO(b"This is not an image file.")
    files = {"file": ("invalid.txt", invalid_file_content, "text/plain")}
    response = test_client.post("/predict", files=files)
    assert response.status_code == 400, "Expected 400 when sending an invalid file."
    detail = response.json().get("detail", "")
    assert "Invalid image file" in detail, "Expected 'Invalid image file' in error detail."
