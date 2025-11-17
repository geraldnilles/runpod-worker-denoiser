import runpod
import torch
import numpy as np
import cv2
from PIL import Image
import base64
import io

# RealESRGAN imports
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# --- CONFIGURATION ---
MODEL_PATH = "./1x_NoiseToner-Poisson-Detailed_108000_G.pth"
TILE_SIZE = 224
TILE_PAD = 16
PRE_PAD = 0
FP16 = True
GPU_ID = 0 if torch.cuda.is_available() else None

# --- GLOBAL MODEL LOADER (Runs once on container start) ---
print("🚀 Loading RealESRGAN model...")

# 1. Define the Architecture
# Note: Ensure these parameters match your specific .pth file. 
# The standard x4plus model is: num_block=23, num_feat=64. 
# If using a custom 1x restoration model, these might differ.
model_arch = RRDBNet(
    num_in_ch=3, 
    num_out_ch=3, 
    num_feat=64, 
    num_block=23, 
    num_grow_ch=32, 
    scale=1 # 1x based on your filename
)

# 2. Instantiate the RealESRGANer wrapper
# This handles tiling, pre-padding, and device management automatically.
upsampler = RealESRGANer(
    scale=1, # Net scale
    model_path=MODEL_PATH,
    model=model_arch,
    tile=TILE_SIZE,
    tile_pad=TILE_PAD,
    pre_pad=PRE_PAD,
    half=FP16,
    gpu_id=GPU_ID
)

print(f"✅ Model loaded. Tiling set to: {TILE_SIZE}")


def decode_image(request_input: dict):
    """
    Decodes base64 to a PIL Image (to extract metadata) 
    and then converts to the OpenCV BGR format required by RealESRGAN.
    """
    try:
        base64_string = request_input["image"]
        img_bytes = base64.b64decode(base64_string)
        img_buffer = io.BytesIO(img_bytes)
        
        # Open in PIL to grab metadata
        pil_img = Image.open(img_buffer)
        
        metadata = {
            "exif": pil_img.info.get("exif"),
            "icc_profile": pil_img.info.get("icc_profile")
        }
        
        # Convert to RGB
        pil_img = pil_img.convert("RGB")
        
        # Convert to Numpy array (RGB)
        img_np = np.array(pil_img)
        
        # Convert RGB to BGR (OpenCV format expected by RealESRGAN)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        return img_bgr, metadata

    except Exception as e:
        print(f"❌ Error decoding image: {e}")
        raise

def encode_image(img_bgr: np.ndarray, metadata: dict = None, quality: int = 95) -> str:
    """
    Converts OpenCV BGR image back to Base64 JPEG with metadata injection.
    """
    # Convert BGR (OpenCV) back to RGB (PIL)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(img_rgb)
    
    buffer = io.BytesIO()
    
    save_kwargs = {
        "format": "JPEG",
        "quality": quality
    }
    
    if metadata:
        if metadata.get("exif"):
            save_kwargs["exif"] = metadata["exif"]
        if metadata.get("icc_profile"):
            save_kwargs["icc_profile"] = metadata["icc_profile"]
            
    pil_img.save(buffer, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def handler(job):
    """
    RunPod Handler Function
    """
    job_input = job["input"]
    
    # 1. Decode
    img_input, metadata = decode_image(job_input)
    
    # 2. Inference (Handles tiling internally)
    # outscale=1 ensures we keep original resolution if it's a restoration model
    try:
        output_img, _ = upsampler.enhance(img_input, outscale=1)
    except RuntimeError as e:
        return {"error": f"RuntimeError during inference: {e}"}
    
    # 3. Encode
    base64_output = encode_image(output_img, metadata=metadata)
    
    return {
        "images": base64_output
    }

if __name__ == "__main__":
    print("🎯 Starting RealESRGAN Handler")
    runpod.serverless.start({"handler": handler})
