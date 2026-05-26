import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
from fashn_vton import TryOnPipeline

app = FastAPI(title="Fashn VTON 1.5 API")

pipeline = None

@app.on_event("startup")
def load_model():
    global pipeline
    # The Docker build will have placed weights here
    weights_dir = "./weights"
    
    # HF_TOKEN is injected via your Cloud Run Secret Manager/Environment Variable
    hf_token = os.getenv("HF_TOKEN")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading pipeline on {device}...")
    
    # Initialize with token for potential gated access
    pipeline = TryOnPipeline(weights_dir=weights_dir, token=hf_token)
    pipeline.to(device)
    print("Pipeline ready.")

@app.post("/try-on")
async def try_on(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("tops"),
    garment_photo_type: str = Form("model"),
):
    try:
        person_pil = Image.open(await person_image.read()).convert("RGB")
        garment_pil = Image.open(await garment_image.read()).convert("RGB")
        
        result = pipeline(
            person_image=person_pil,
            garment_image=garment_pil,
            category=category,
            garment_photo_type=garment_photo_type,
            num_timesteps=30,
            guidance_scale=1.5,
            segmentation_free=True
        )
        
        buf = BytesIO()
        result.images[0].save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
