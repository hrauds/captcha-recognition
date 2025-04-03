from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch

from app.preprocessing import get_transform
from app.decoding import decode_predictions
from app.model import IDX_TO_CHAR

router = APIRouter()

@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file provided: {e}")

    transform = get_transform()
    try:
        input_tensor = transform(image).unsqueeze(0)
        print(f"Input tensor shape: {input_tensor.shape}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    try:
        model = request.app.state.model
        with torch.no_grad():
            outputs = model(input_tensor)
        print(f"Outputs shape: {outputs.shape}")
        prediction = decode_predictions(outputs, IDX_TO_CHAR)[0]
        print(f"Prediction: {prediction}")
    except Exception as e:
        print("Error during model inference:", str(e))
        raise HTTPException(status_code=500, detail=f"Error during model inference: {e}")

    return JSONResponse(content={"prediction": prediction})
