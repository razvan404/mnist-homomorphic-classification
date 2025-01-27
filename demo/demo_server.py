from typing import List

import torch
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from src.ckks import Encryptor, CkksCompatibleMnistClassifier
from src.classifier import MnistClassifier


app = FastAPI()
raw_model = MnistClassifier()
raw_model.load_state_dict(torch.load("models/mnist_classifier.pth", weights_only=True))
raw_model.eval()
ckks_model = CkksCompatibleMnistClassifier(raw_model)


class InferenceEncryptor(BaseModel):
    context: str
    windows_nb: int


class InferenceRequest(BaseModel):
    encryptor: InferenceEncryptor
    image: str


class InferenceResponse(BaseModel):
    preds: str


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    try:
        serialized_encryptor = {
            "context": request.encryptor.context,
            "windows_nb": request.encryptor.windows_nb,
        }
        encryptor = Encryptor.deserialize(serialized_encryptor)
        print(f"{encryptor.has_secret_key() = }")

        serialized_image = request.image
        image = encryptor.deserialize_data(serialized_image)

        preds = ckks_model(image)
        serialized_preds = encryptor.serialize_data(preds)

        return InferenceResponse(preds=str(serialized_preds))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
