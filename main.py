from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import io
import os
from datetime import datetime
from ocr_utils import preprocess_image_all_rotations, extract_ocr_data_robust, detect_pii, mask_pii
from models import save_upload_info
from PIL import Image
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "static/uploads"
MASKED_DIR = "static/masked"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MASKED_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_image(
    request: Request, 
    file: UploadFile = File(...),
    mask_type: str = Form("black")
):
    try:
        # Validate mask type
        valid_mask_types = ["black", "blur", "pixelate"]
        if mask_type not in valid_mask_types:
            mask_type = "black"
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Save original
        orig_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(orig_path, "wb") as f:
            f.write(image_bytes)
        
        # Robust preprocessing (all rotations and modes)
        preprocessed_images = preprocess_image_all_rotations(image_bytes)
        # Robust OCR
        ocr_results = extract_ocr_data_robust(preprocessed_images)
        # Enhanced PII detection
        pii_boxes = detect_pii(ocr_results)
        # Use the first preprocessed image for masking
        mask_img = preprocessed_images[0]
        # Apply the selected masking type
        if mask_type == "pixelate":
            from ocr_utils import pixelate_mask_pii
            masked_img = pixelate_mask_pii(mask_img, pii_boxes)
        else:
            masked_img = mask_pii(mask_img, pii_boxes, mask_type)
        
        # Save masked image with mask type in filename
        base_name, ext = os.path.splitext(file.filename)
        masked_filename = f"{base_name}_{mask_type}{ext}"
        masked_path = os.path.join(MASKED_DIR, masked_filename)
        
        masked_pil = Image.fromarray(masked_img)
        masked_pil.save(masked_path)
        
        # Save to MongoDB
        try:
            await save_upload_info(masked_filename, pii_boxes, datetime.utcnow())
        except Exception as e:
            print(f"MongoDB save failed: {e}")  # Continue even if MongoDB fails
        
        # Return both images in HTML
        orig_url = f"/static/uploads/{file.filename}"
        masked_url = f"/static/masked/{masked_filename}"
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "orig_url": orig_url,
            "masked_url": masked_url,
            "pii": pii_boxes,
            "ocr_count": len(ocr_results),
            "mask_type": mask_type
        })
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Error processing image: {str(e)}"
        }) 