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
    weights_dir = "./weights"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading pipeline on {device}...")
    
    pipeline = TryOnPipeline(weights_dir=weights_dir)
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
        
        # 1. Disable gradient calculation to save massive amounts of VRAM
        with torch.inference_mode(): 
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
        
    finally:
        # 2. Aggressively clear the GPU cache after every request
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
