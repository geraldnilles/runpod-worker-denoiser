import runpod

import torch
from PIL import Image
import numpy as np
from spandrel import ModelLoader

import base64
import io
import os
import subprocess
import tempfile

def load_image(request_input: dict): # -> tuple[torch.Tensor, dict]:
    """
    Loads an image from a base64 string and converts it to a (B, C, H, W) tensor.
    Also extracts metadata (Exif, ICC Profile).
    """
    
    # 1. Extract the base64 string from the input dictionary
    try:
        base64_string = request_input["image"]
    except KeyError:
        print("❌ Error: 'image' key not found in request_input dictionary.")
        raise
    except TypeError:
        print(f"❌ Error: request_input was not a dictionary. Got {type(request_input)} instead.")
        raise

    # 2. Decode the base64 string into bytes
    try:
        img_bytes = base64.b64decode(base64_string)
    except Exception as e:
        print(f"❌ Error decoding base64 string: {e}")
        raise

    # 3. Create an in-memory file-like object from the bytes
    img_buffer = io.BytesIO(img_bytes)

    # 4. Open the image using PIL
    img = Image.open(img_buffer)
    
    # ### NEW: Extract Metadata before converting ###
    # We capture the raw bytes for Exif and ICC Profile. 
    # Note: PIL does not always expose XMP easily, but Exif usually contains the bulk of the data.
    metadata = {
        "exif": img.info.get("exif"),
        "icc_profile": img.info.get("icc_profile")
    }
    
    # Convert to RGB (This strips metadata from the object, but we saved it above)
    img = img.convert("RGB")

    # 5. Convert to numpy array (H, W, C)
    img_np = np.array(img)

    # 6. Convert to torch tensor (C, H, W) and scale to [0, 1]
    # Note: We keep this as float32 on CPU for safe division, we cast to half in main()
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0

    # 7. Add batch dimension (B, C, H, W) and RETURN METADATA
    return img_tensor.unsqueeze(0), metadata

def save_image(tensor: torch.Tensor, metadata: dict = None, quality: int = 90) -> str:
    """Converts a (B, C, H, W) tensor to a base64 encoded AVIF string using avifenc."""
    
    # Cast back to float32 for saving
    tensor = tensor.float()

    # Remove batch dimension (C, H, W)
    tensor = tensor.squeeze(0)

    # Clamp values to [0, 1] just in case
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to (H, W, C) numpy array and scale to [0, 255]
    img_np = (tensor.permute(1, 2, 0) * 255.0).byte().cpu().numpy()

    # Convert to PIL Image
    img = Image.fromarray(img_np)

    # Create temporary files for PNG and AVIF
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as png_file:
        png_path = png_file.name
    
    with tempfile.NamedTemporaryFile(suffix='.avif', delete=False) as avif_file:
        avif_path = avif_file.name

    try:
        # Save as PNG first
        img.save(png_path, format='PNG')
        print(f"✅ Saved image as PNG: {png_path}")

        # Convert to AVIF using avifenc
        cmd = ['avifenc', f'-q {quality}', png_path, avif_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"avifenc failed: {result.stderr}")
        
        print(f"✅ Converted to AVIF: {avif_path}")

        # Read AVIF file and encode to base64
        with open(avif_path, 'rb') as f:
            avif_bytes = f.read()
        
        base64_string = base64.b64encode(avif_bytes).decode('utf-8')
        print(f"✅ Successfully encoded AVIF to base64")

        return base64_string

    finally:
        # Clean up temporary files
        if os.path.exists(png_path):
            os.unlink(png_path)
        if os.path.exists(avif_path):
            os.unlink(avif_path)

def split_image_into_patches(image_tensor: torch.Tensor, patch_size: int, overlap: int):
    """
    Splits a large input image tensor into a series of patches with overlap.
    """
    _, _, H, W = image_tensor.shape

    stride = patch_size - overlap
    patches_with_coords = []

    y_starts = [i * stride for i in range((H - overlap) // stride)]
    if (H - patch_size) % stride != 0 or H < patch_size:
        y_starts.append(H - patch_size)
    y_starts = sorted(list(set(y_starts))) 

    x_starts = [i * stride for i in range((W - overlap) // stride)]
    if (W - patch_size) % stride != 0 or W < patch_size: 
        x_starts.append(W - patch_size)
    x_starts = sorted(list(set(x_starts))) 

    for y_start in y_starts:
        for x_start in x_starts:
            y_end = y_start + patch_size
            x_end = x_start + patch_size

            patch_tensor = image_tensor[:, :, y_start:y_end, x_start:x_end]
            patches_with_coords.append((patch_tensor, (y_start, y_end, x_start, x_end)))

    return patches_with_coords

def stitch_patches_together(processed_patches: list, original_H: int, original_W: int, patch_size: int, overlap: int) -> torch.Tensor:
    """
    Reassembles processed patches back into a single, large image, handling overlaps.
    """
    if not processed_patches:
        raise ValueError("No processed patches provided.")

    # Check dtype from the first patch to ensure canvas matches (float16 vs float32)
    ref_tensor = processed_patches[0][0]
    B, C, _, _ = ref_tensor.shape
    dtype = ref_tensor.dtype
    device = ref_tensor.device

    # ### CHANGED: Initialize canvas with correct dtype ###
    stitched_image = torch.zeros((B, C, original_H, original_W), device=device, dtype=dtype)
    weight_map = torch.zeros((B, 1, original_H, original_W), device=device, dtype=dtype) 

    blend_window = torch.ones((1, 1, patch_size, patch_size), device=device, dtype=dtype)

    for patch_tensor, (y_start, y_end, x_start, x_end) in processed_patches:
        if patch_tensor.dim() == 3:
            patch_tensor = patch_tensor.unsqueeze(0) 

        stitched_image[:, :, y_start:y_end, x_start:x_end] += patch_tensor
        weight_map[:, :, y_start:y_end, x_start:x_end] += blend_window

    stitched_image = stitched_image / (weight_map + 1e-6)

    return stitched_image

def main(request_input):

    # --- 1. Setup Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using CUDA (GPU)")
    else:
        device = torch.device("cpu")
        print("🐌 Using CPU")

    # --- 2. Load Model using Spandrel ---
    print(f"Loading model ...")
    try:
        # Ensure this path matches your RunPod volume path
        state_dict = torch.load("./1x_NoiseToner-Poisson-Detailed_108000_G.pth", map_location="cpu")
        loader = ModelLoader()
        model = loader.load_from_state_dict(state_dict)

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please ensure your .pth file is a valid NAFNet state_dict.")
        return

    model.eval()
    # ### CHANGED: Move model to device AND cast to half precision ###
    model = model.to(device).half()
    print(f"👍 Model loaded successfully: {model.__class__.__name__} (fp16)")

    # --- 3. Load and Prepare Image ---
    print(f"Loading image from ...")
    try:
        # ### NEW: Receive Metadata tuple ###
        input_tensor, original_metadata = load_image(request_input)
        
        # ### CHANGED: Move input to device AND cast to half precision ###
        input_tensor = input_tensor.to(device).half()
        _, _, original_H, original_W = input_tensor.shape
    except FileNotFoundError:
        print(f"❌ Error: Input file not found")
        return
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # --- Tiling Strategy Constants ---
    PATCH_SIZE = 512
    OVERLAP = 32 
    BATCH_SIZE = 16  
    
    # --- 4. Split Image into Patches ---
    print(f"Splitting image into {PATCH_SIZE}x{PATCH_SIZE} patches with {OVERLAP} overlap...")
    patches_with_coords = split_image_into_patches(input_tensor, PATCH_SIZE, OVERLAP)
    num_patches_total = len(patches_with_coords)
    print(f"Generated {num_patches_total} patches.")

    # --- 5. Run Denoising (Inference) on Patches (BATCHED & GPU RESIDENT) ---
    print(f"Running denoising on patches in batches of {BATCH_SIZE}...")
    processed_patches_with_coords = []

    with torch.no_grad():
        for i in range(0, num_patches_total, BATCH_SIZE):

            batch_data = patches_with_coords[i : i + BATCH_SIZE]

            batch_patches_list = [item[0] for item in batch_data]
            batch_coords_list = [item[1] for item in batch_data]

            batch_input = torch.cat(batch_patches_list, dim=0)
            
            # Model is fp16, Input is fp16 -> Output will be fp16
            output_batch = model(batch_input)
            split_output_patches = output_batch.split(1, dim=0)

            for output_patch, coords in zip(split_output_patches, batch_coords_list):
                processed_patches_with_coords.append((output_patch, coords))

            if True:
                print(f"  Processed {i + len(batch_data)}/{num_patches_total} patches...")

    # --- 6. Stitch Patches Together (ON GPU) ---
    print("Stitching processed patches together on GPU...")

    final_output_tensor_gpu = stitch_patches_together(
        processed_patches_with_coords,
        original_H,
        original_W,
        PATCH_SIZE,
        OVERLAP
    )

    # --- ONLY NOW do we move to CPU ---
    print("Moving final image to CPU...")
    final_output_tensor = final_output_tensor_gpu.cpu()

    # --- 7. Encode Output Image to Base64 ---
    base64_image_string = "" 
    try:
        # ### NEW: Pass metadata to save function ###
        base64_image_string = save_image(final_output_tensor, metadata=original_metadata)
    except Exception as e:
        print(f"❌ Error encoding image: {e}")

    return base64_image_string, len(patches_with_coords)

def handler(job):
    
    image_string, num_patches = main(job["input"])
    
    return { "images": image_string,        
             "num_patches": num_patches
           }

if __name__ == "__main__":
    print("🎯 Starting Deblur Handler")
    runpod.serverless.start({"handler": handler})
