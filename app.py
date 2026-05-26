import io
import os
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from fashn_vton import TryOnPipeline

pipeline = None

WEIGHTS_DIR = "/app/weights"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # Download weights if missing
    model_file = os.path.join(WEIGHTS_DIR, "model.safetensors")

    if not os.path.exists(model_file):
        print("Downloading model weights...")

        subprocess.run(
            [
                "python3",
                "/app/fashn-vton-1.5/scripts/download_weights.py",
                "--weights-dir",
                WEIGHTS_DIR,
            ],
            check=True,
        )

    print("Loading FASHN VTON pipeline...")

    pipeline = TryOnPipeline(
        weights_dir=WEIGHTS_DIR,
        device="cuda"
    )

    print("Pipeline loaded successfully.")

    yield


app = FastAPI(
    title="FASHN VTON API",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {
        "message": "FASHN VTON API running"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }


@app.post("/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form(...),
    garment_photo_type: str = Form("flat-lay"),
    num_samples: int = Form(1),
    num_timesteps: int = Form(30),
    guidance_scale: float = Form(1.5),
    seed: int = Form(42),
    segmentation_free: bool = Form(True),
):
    try:
        person_bytes = await person_image.read()
        garment_bytes = await garment_image.read()

        person = Image.open(
            io.BytesIO(person_bytes)
        ).convert("RGB")

        garment = Image.open(
            io.BytesIO(garment_bytes)
        ).convert("RGB")

        result = pipeline(
            person_image=person,
            garment_image=garment,
            category=category,
            garment_photo_type=garment_photo_type,
            num_samples=num_samples,
            num_timesteps=num_timesteps,
            guidance_scale=guidance_scale,
            seed=seed,
            segmentation_free=segmentation_free,
        )

        image = result.images[0]

        output = io.BytesIO()
        image.save(output, format="PNG")
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="image/png"
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
