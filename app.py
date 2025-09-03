import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from visualize import visualize
from typing import Optional
import cv2
import time
import numpy as np
from ocr_engine import OcrEngine
import config as cfg
from pdf2image import convert_from_bytes
import onnxruntime

providers = onnxruntime.get_available_providers()
print(providers)

ai_server = OcrEngine(cfg.config_detector, cfg.config_recognizer, cfg.config_doclayout)
app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.post("/ocr")
async def text_detection(
    file: UploadFile = File(...),
    ocr_det_min_size: float = Form(720),
    ocr_det_binary_threshold: float = Form(0.1),
    ocr_det_polygon_threshold: float = Form(0.1),
    ocr_det_batch_size: int = Form(4),
    ocr_rec_image_width: int = Form(240),
    ocr_rec_batch_size: int = Form(32),
    pdf_dpi: int = Form(120),
    pdf_thread_count: int = Form(4)
):
    extension = file.filename.split(".")[-1]
    if extension.lower() not in ["jpeg", "jpg", "jpe", "webp", "jp2", "png", "pdf"]:
        return "File must be jpg, png or pdf format!"

    result = {}
    if extension.lower() == "pdf":
        st = time.time()
        pages = convert_from_bytes(await file.read(), dpi=pdf_dpi, thread_count=pdf_thread_count)
        print("time process files ", time.time() - st)
        list_images = []
        st = time.time()
        for i, page in enumerate(pages):
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            list_images.append(opencv_image)
        result = ai_server.run(
            list_images,
            ocr_det_min_size,
            ocr_det_binary_threshold,
            ocr_det_polygon_threshold,
            ocr_det_batch_size,
            ocr_rec_image_width,
            ocr_rec_batch_size
        )
        print("time process ocr ", time.time() - st)
            
    else:
        # Read files
        st = time.time()
        image_bytes = await file.read()
    
        # Convert byte data to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode the numpy array to OpenCV image
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        print("time process files ", time.time() - st)
        st = time.time()
        result = ai_server.run(
            [img],
            ocr_det_min_size,
            ocr_det_binary_threshold,
            ocr_det_polygon_threshold,
            ocr_det_batch_size,
            ocr_rec_image_width,
            ocr_rec_batch_size
        )
        print("time process ocr ", time.time() - st)
    json_result = {"result": result}
    return json_result

@app.post("/layout_analysis")
async def layout_analysis(
    file: UploadFile = File(...),
    ocr_det_min_size: float = Form(720),
    ocr_det_binary_threshold: float = Form(0.1),
    ocr_det_polygon_threshold: float = Form(0.1),
    ocr_det_batch_size: int = Form(4),
    ocr_rec_image_width: int = Form(240),
    ocr_rec_batch_size: int = Form(32),
    pdf_dpi: int = Form(120),
    pdf_thread_count: int = Form(4)
    ):
    extension = file.filename.split(".")[-1]
    if extension.lower() not in ["jpeg", "jpg", "jpe", "webp", "jp2", "png", "pdf"]:
        return "File must be jpg, png or pdf format!"

    result = {}
    if extension.lower() == "pdf":
        st = time.time()
        pages = convert_from_bytes(await file.read(), dpi=pdf_dpi, thread_count=pdf_thread_count)
        print("time process files ", time.time() - st)
        list_images = []
        st = time.time()
        for i, page in enumerate(pages):
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            list_images.append(opencv_image)
        result = ai_server.run_layout_analysis(
            list_images,
            ocr_det_min_size,
            ocr_det_binary_threshold,
            ocr_det_polygon_threshold,
            ocr_det_batch_size,
            ocr_rec_image_width,
            ocr_rec_batch_size
        )
        print("time process layout analysis ", time.time() - st)
            
    else:
        # Read files
        st = time.time()
        image_bytes = await file.read()
    
        # Convert byte data to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode the numpy array to OpenCV image
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        print("time process files ", time.time() - st)
        st = time.time()
        result = ai_server.run_layout_analysis(
            [img],
            ocr_det_min_size,
            ocr_det_binary_threshold,
            ocr_det_polygon_threshold,
            ocr_det_batch_size,
            ocr_rec_image_width,
            ocr_rec_batch_size
        )
        print("time process layout analysis ", time.time() - st)
    json_result = {"result": result}
    return json_result
