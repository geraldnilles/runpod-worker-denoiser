import runpod
import torch
from PIL import Image, ImageOps
import numpy as np
from spandrel import ModelLoader
import base64
import io
import os

# --- Global Initialization (Runs once when worker starts) ---
print("🚀 Worker starting... Initializing model.")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ Device: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ Device: CPU (Performance will be low)")

# Enable CUDNN benchmarks for consistent input sizes
torch.backends.cudnn.benchmark = True

MODEL = None

try:
    # Load model once at startup
    # Ensure this path matches your RunPod volume path
    MODEL_PATH = "./1x_NoiseToner-Poisson-Detailed_108000_G.pth"
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        loader = ModelLoader()
        MODEL = loader.load_from_state_dict(state_dict)
        MODEL.eval()
        MODEL = MODEL.to(DEVICE).half() # Cast to half immediately
        
        # Optimize with Torch Compile (Great for L4 GPU)
        # mode="reduce-overhead" is great for many small patch inferences
        print("⚙️ Compiling model with torch.compile...")
        try:
            MODEL = torch.compile(MODEL, mode="reduce-overhead")
        except Exception as e:
            print(f"⚠️ torch.compile failed (ignoring): {e}")
            
        print(f"👍 Model loaded and compiled: {MODEL.__class__.__name__}")
    else:
        print(f"❌ Error: Model file not found at {MODEL_PATH}")

except Exception as e:
    print(f"❌ Critical Error loading model: {e}")

# --- Helper Functions ---

def load_image(request_input: dict):
    """Loads image from base64, returns (1, C, H, W) float16 tensor on GPU."""
    try:
        base64_string = request_input["image"]
        img_bytes = base64.b64decode(base64_string)
        img_buffer = io.BytesIO(img_bytes)
        img = Image.open(img_buffer).convert("RGB")
        img = ImageOps.exif_transpose(img)
        
        # Convert directly to tensor from numpy to avoid intermediate copies if possible
        img_np = np.array(img)
        
        # To Device -> Float -> Permute -> Div -> Half -> Unsqueeze
        # We do this sequence to ensure we don't consume excessive VRAM with float32
        img_tensor = torch.from_numpy(img_np).to(DEVICE).float().permute(2, 0, 1) / 255.0
        
        return img_tensor.unsqueeze(0).half()

    except Exception as e:
        print(f"❌ Error in load_image: {e}")
        raise

def save_image(tensor: torch.Tensor) -> str:
    """Converts (C, H, W) tensor to base64 string."""
    # Keep on GPU for clamping and multiplication, then move to CPU
    tensor = torch.clamp(tensor, 0, 1)
    img_np = (tensor.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
    
    img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=False) # optimize=False is faster
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def split_image_into_patches(image_tensor: torch.Tensor, patch_size: int, overlap: int):
    _, _, H, W = image_tensor.shape
    stride = patch_size - overlap
    
    # Calculate grid
    y_starts = list(range(0, H - overlap, stride))
    if (H - patch_size) % stride != 0 or H < patch_size:
        y_starts.append(H - patch_size)
    # Ensure unique and sorted, though range usually handles this
    y_starts = sorted(list(set(y_starts)))

    x_starts = list(range(0, W - overlap, stride))
    if (W - patch_size) % stride != 0 or W < patch_size:
        x_starts.append(W - patch_size)
    x_starts = sorted(list(set(x_starts)))

    patches = []
    coords = []

    for y_start in y_starts:
        for x_start in x_starts:
            y_end = y_start + patch_size
            x_end = x_start + patch_size
            
            # Slicing is a view, very cheap
            patches.append(image_tensor[:, :, y_start:y_end, x_start:x_end])
            coords.append((y_start, y_end, x_start, x_end))

    return patches, coords

def stitch_patches_together(processed_patches_list, coords_list, original_H, original_W, patch_size, overlap):
    if not processed_patches_list:
        raise ValueError("No patches to stitch.")

    # Assume all patches have same shape/dtype/device as the first
    ref = processed_patches_list[0]
    dtype = ref.dtype
    
    # Pre-allocate output canvas
    stitched_image = torch.zeros((1, 3, original_H, original_W), device=DEVICE, dtype=dtype)
    weight_map = torch.zeros((1, 1, original_H, original_W), device=DEVICE, dtype=dtype)
    
    # Create blend window once
    blend_window = torch.ones((1, 1, patch_size, patch_size), device=DEVICE, dtype=dtype)

    for patch, (y_start, y_end, x_start, x_end) in zip(processed_patches_list, coords_list):
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)
            
        stitched_image[:, :, y_start:y_end, x_start:x_end] += patch
        weight_map[:, :, y_start:y_end, x_start:x_end] += blend_window

    stitched_image /= (weight_map + 1e-6)
    return stitched_image

@torch.inference_mode()
def process_job(job_input):
    if MODEL is None:
        return {"error": "Model not loaded correctly."}

    # --- Configuration ---
    # L4 can handle larger batches. 
    # If OOM occurs, reduce BATCH_SIZE to 16.
    BATCH_SIZE = 16
    PATCH_SIZE = 512
    OVERLAP = 32

    # --- Load ---
    input_tensor = load_image(job_input)
    _, _, original_H, original_W = input_tensor.shape
    
    # --- Split ---
    # Note: We return lists of tensors, not coords attached to tensors, for cleaner batching
    patches_list, coords_list = split_image_into_patches(input_tensor, PATCH_SIZE, OVERLAP)
    num_patches = len(patches_list)
    print(f"🧩 Processing {num_patches} patches (Batch Size: {BATCH_SIZE})...")

    processed_patches = []

    # --- Batch Inference ---
    # We iterate by index to slice the list
    for i in range(0, num_patches, BATCH_SIZE):
        batch_input_tensors = patches_list[i : i + BATCH_SIZE]
        
        # Stack into a single tensor (B, C, H, W)
        batch_stack = torch.cat(batch_input_tensors, dim=0)
        
        # Inference
        output_batch = MODEL(batch_stack)
        
        # Split back into list of tensors
        # split(1) returns tuple of (1, C, H, W) tensors
        processed_patches.extend([p for p in output_batch.split(1, dim=0)])

    # --- Stitch ---
    final_output = stitch_patches_together(
        processed_patches, 
        coords_list, 
        original_H, 
        original_W, 
        PATCH_SIZE, 
        OVERLAP
    )

    # --- Encode ---
    # Squeeze batch dim before saving
    final_b64 = save_image(final_output.squeeze(0))
    
    return final_b64, num_patches

def handler(job):
    try:
        image_string, num_patches = process_job(job["input"])
        return {
            "images": image_string,
            "num_patches": num_patches
        }
    except Exception as e:
        print(f"❌ Job failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    print("🎯 Starting Optimized Handler")
    runpod.serverless.start({"handler": handler})
