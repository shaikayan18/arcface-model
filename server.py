from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import uvicorn

from attendance_backend import ArcFaceAttendanceBackend

app = FastAPI(title="ArcFace Attendance System", version="1.0.0")

def get_client_ip(request: Request) -> str:
    """Extract client IP address from request"""
    # Check for forwarded IP (from proxy/ngrok)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    # Check for real IP (from proxy)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to direct client IP
    return request.client.host if request.client else "unknown"

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add OPTIONS handler for preflight requests
@app.options("/{path:path}")
async def options_handler(path: str):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Initialize backend
backend = ArcFaceAttendanceBackend(threshold=0.8)

def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
        return image_np
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

@app.post("/register")
async def register_person(data: dict, request: Request):
    """Register a new person with their face"""
    try:
        name = data.get('name')
        image_data = data.get('image')
        
        if not name or not image_data:
            raise HTTPException(status_code=400, detail="Name and image are required")
        
        # Get client IP
        client_ip = get_client_ip(request)
        
        # Decode image
        image = decode_base64_image(image_data)
        
        # Register person
        success, message = backend.register_person(name, image)
        
        if success:
            # Log IP activity
            backend.log_ip_activity(name, client_ip, "registration")
            return JSONResponse(content={"success": True, "message": message})
        else:
            raise HTTPException(status_code=400, detail=message)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/recognize")
async def recognize_person(data: dict, request: Request):
    """Recognize a person from image"""
    try:
        image_data = data.get('image')
        
        if not image_data:
            raise HTTPException(status_code=400, detail="Image is required")
        
        # Get client IP
        client_ip = get_client_ip(request)
        
        # Decode image
        image = decode_base64_image(image_data)
        
        # Recognize person
        name, distance = backend.recognize_person(image)
        
        # Log IP activity if person recognized
        if name:
            backend.log_ip_activity(name, client_ip, "recognition")
        
        return JSONResponse(content={
            "name": name if name else "Unknown",
            "distance": float(distance),
            "recognized": name is not None
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@app.post("/attendance")
async def log_attendance(data: dict, request: Request):
    """Log attendance for a person"""
    try:
        name = data.get('name')
        
        if not name:
            raise HTTPException(status_code=400, detail="Name is required")
        
        # Get client IP
        client_ip = get_client_ip(request)
        
        # Log attendance
        success, message = backend.log_attendance(name)
        
        # Log IP activity
        backend.log_ip_activity(name, client_ip, "attendance")
        
        return JSONResponse(content={
            "success": success,
            "message": message
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Attendance logging failed: {str(e)}")

@app.get("/persons")
async def get_registered_persons():
    """Get list of registered persons"""
    try:
        persons = backend.get_registered_persons()
        return JSONResponse(content={"persons": persons})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get persons: {str(e)}")

@app.get("/attendance")
async def get_attendance_records():
    """Get attendance records"""
    try:
        records = backend.get_attendance_records()
        return JSONResponse(content={"records": records})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get attendance: {str(e)}")

@app.get("/ip-records")
async def get_ip_records():
    """Get IP tracking records"""
    try:
        records = backend.get_ip_records()
        return JSONResponse(content={"records": records})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get IP records: {str(e)}")

@app.get("/")
async def root():
    return {"message": "ArcFace Attendance System API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
