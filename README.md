# CAPTCHA Recognition API

This repository contains an improved CAPTCHA recognition model and a FastAPI service for inference.

## Model Improvement

* **Improvements:**
   * **Preprocessing:** cropped 3 pixels from the left/right to clean the images without losing detail.
   * **Model architecture:** added CNN layers with dropout and a bidirectional LSTM. Added OneCycleLR scheduler for training.
* **Results:**
   * **Baseline accuracy:** ~40–60%
   * **Improved accuracy:** >99%

## API & Containerization

* **FastAPI Implementation:** A `POST /predict` endpoint accepts an image file and returns the predicted CAPTCHA text. Basic error handling and input validation are implemented.
* **Model Serving:** The model is loaded once during FastAPI's startup using a lifespan event.
* **API Docs:** Auto-generated documentation is available at `/docs`.

## Setup Instructions

1. **Clone the Repository:**

```bash
git clone https://github.com/hrauds/captcha-recognition.git
```

2. **Build and Run the Container:**

```bash
docker-compose up --build
```

The API will be available at http://localhost:8000.

3. **Example API Call:**

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path_to_image.png"
```

4. **Run Tests:**

```bash
pytest tests/test_api.py
```

## Experience
I really liked the exercise, especially the model training part, it was fun trying out different approaches. I spent quite a bit of time trying to remove the line and even though I didn’t end up using that method, it was a great challenge and I learned a lot from it. I haven’t done much image processing before, so it was really interesting to read about and explore.