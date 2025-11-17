import runpod

import torch
from PIL import Image
import numpy as np
from spandrel import ModelLoader

import base64
import io
import os

def load_image(request_input: dict) -> torch.Tensor:
    """Loads an image from a base64 string in the input dict and converts it to a (B, C, H, W) tensor."""
    
    # 1. Extract the base64 string from the input dictionary
    #    We assume the string is under the key 'image'.
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

    # 4. Open the image using PIL and convert to RGB
    img = Image.open(img_buffer).convert("RGB")

    # 5. Convert to numpy array (H, W, C)
    img_np = np.array(img)

    # 6. Convert to torch tensor (C, H, W) and scale to [0, 1]
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0

    # 7. Add batch dimension (B, C, H, W)
    return img_tensor.unsqueeze(0)

def save_image(tensor: torch.Tensor, quality: int = 95) -> str:
    """Converts a (B, C, H, W) tensor to a base64 encoded JPEG string."""
    # Remove batch dimension (C, H, W)
    tensor = tensor.squeeze(0)

    # Clamp values to [0, 1] just in case
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to (H, W, C) numpy array and scale to [0, 255]
    img_np = (tensor.permute(1, 2, 0) * 255.0).byte().cpu().numpy()

    # Convert to PIL Image
    img = Image.fromarray(img_np)

    # Create an in-memory buffer
    buffer = io.BytesIO()

    # Save image to buffer as JPEG
    img.save(buffer, format="JPEG", quality=quality)

    # Get the bytes from the buffer
    img_bytes = buffer.getvalue()

    # Encode bytes to base64 string
    base64_string = base64.b64encode(img_bytes).decode('utf-8')

    print(f"✅ Successfully encoded image to base64.")

    return base64_string

def split_image_into_patches(image_tensor: torch.Tensor, patch_size: int, overlap: int):
    """
    Splits a large input image tensor into a series of patches with overlap.
    Ensures all patches align perfectly with the original image's edges without requiring any padding.

    Args:
        image_tensor (torch.Tensor): The input image tensor (B, C, H, W).
        patch_size (int): The desired size of each square patch (e.g., 512).
        overlap (int): The approximate overlap between patches (e.g., 64).

    Returns:
        list: A list of tuples, where each tuple contains (patch_tensor, (y_start, y_end, x_start, x_end)).
    """
    _, _, H, W = image_tensor.shape

    # Calculate the stride for both height and width
    stride = patch_size - overlap

    patches_with_coords = []

    # Determine the number of patches needed along the height and width
    # Ensure the last patch perfectly covers the image's edge

    # Calculate the number of patches by ensuring the last patch starts at H - patch_size
    # This ensures exact alignment with the image edge for the last patch.
    y_starts = [i * stride for i in range((H - overlap) // stride)]
    if (H - patch_size) % stride != 0 or H < patch_size: # Add the last patch if not already covered or if image is smaller than patch
        y_starts.append(H - patch_size)
    y_starts = sorted(list(set(y_starts))) # Remove duplicates and sort

    x_starts = [i * stride for i in range((W - overlap) // stride)]
    if (W - patch_size) % stride != 0 or W < patch_size: # Add the last patch if not already covered or if image is smaller than patch
        x_starts.append(W - patch_size)
    x_starts = sorted(list(set(x_starts))) # Remove duplicates and sort

    for y_start in y_starts:
        for x_start in x_starts:
            y_end = y_start + patch_size
            x_end = x_start + patch_size

            # Extract the patch
            patch_tensor = image_tensor[:, :, y_start:y_end, x_start:x_end]
            patches_with_coords.append((patch_tensor, (y_start, y_end, x_start, x_end)))

    return patches_with_coords

def stitch_patches_together(processed_patches: list, original_H: int, original_W: int, patch_size: int, overlap: int) -> torch.Tensor:
    """
    Reassembles processed patches back into a single, large image, handling overlaps.

    Args:
        processed_patches (list): A list of tuples, where each tuple contains
                                    (processed_patch_tensor, (y_start, y_end, x_start, x_end)).
                                    The processed_patch_tensor is (B, C, patch_H, patch_W).
        original_H (int): Original height of the image.
        original_W (int): Original width of the image.
        patch_size (int): The size of each square patch.
        overlap (int): The approximate overlap between patches.

    Returns:
        torch.Tensor: The reassembled, seamless output image (B, C, original_H, original_W).
    """
    if not processed_patches:
        raise ValueError("No processed patches provided.")

    # Assuming all patches have the same channel count as the first patch
    B, C, _, _ = processed_patches[0][0].shape

    # Initialize an empty tensor for the stitched image and a weight map for blending
    stitched_image = torch.zeros((B, C, original_H, original_W), device=processed_patches[0][0].device)
    weight_map = torch.zeros((B, 1, original_H, original_W), device=processed_patches[0][0].device) # (B, 1, H, W) for broadcasting

    # Create a blending window for smoother transitions in overlap areas
    # For a simple average, we can use a window of ones.
    # For more advanced blending, one could use a linear or cosine fade.
    # For this task, we'll use a simple count (1s) and divide later.
    blend_window = torch.ones((1, 1, patch_size, patch_size), device=processed_patches[0][0].device)

    for patch_tensor, (y_start, y_end, x_start, x_end) in processed_patches:
        # Ensure patch tensor has the correct shape (B, C, H, W)
        if patch_tensor.dim() == 3:
            patch_tensor = patch_tensor.unsqueeze(0) # Add batch dimension if missing

        # Add the patch to the stitched image
        stitched_image[:, :, y_start:y_end, x_start:x_end] += patch_tensor

        # Add to the weight map (counting how many patches contribute to each pixel)
        weight_map[:, :, y_start:y_end, x_start:x_end] += blend_window

    # Divide the stitched image by the weight map to average overlapping regions
    # Handle potential division by zero for any uncovered regions (shouldn't happen with correct splitting)
    # Add a small epsilon to weight_map to avoid division by zero in case of an empty region
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
        state_dict = torch.load("./1x_NoiseToner-Poisson-Detailed_108000_G.pth", map_location="cpu")
        loader = ModelLoader()
        model = loader.load_from_state_dict(state_dict)

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please ensure your .pth file is a valid NAFNet state_dict.")
        return

    model.eval()
    model = model.to(device)
    print(f"👍 Model loaded successfully: {model.__class__.__name__}")

    # --- 3. Load and Prepare Image ---
    print(f"Loading image from ...")
    try:
        # Load the image directly onto the target device
        input_tensor = load_image(request_input).to(device) 
        _, _, original_H, original_W = input_tensor.shape
    except FileNotFoundError:
        print(f"❌ Error: Input file not found")
        return
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # --- Tiling Strategy Constants ---
    PATCH_SIZE = 512 #1024
    OVERLAP = 64 #128
    BATCH_SIZE = 4  # <-- ⭐ YOUR NEW BATCH SIZE
    
    # --- 4. Split Image into Patches ---
    print(f"Splitting image into {PATCH_SIZE}x{PATCH_SIZE} patches with {OVERLAP} overlap...")
    # Note: patches_with_coords contains tensors that are *already on the device*
    # because they are slices of input_tensor.
    patches_with_coords = split_image_into_patches(input_tensor, PATCH_SIZE, OVERLAP)
    num_patches_total = len(patches_with_coords)
    print(f"Generated {num_patches_total} patches.")

    # --- 5. Run Denoising (Inference) on Patches (BATCHED & GPU RESIDENT) ---
    print(f"Running denoising on patches in batches of {BATCH_SIZE}...")
    processed_patches_with_coords = []

    with torch.no_grad():
        for i in range(0, num_patches_total, BATCH_SIZE):

            # 1. Get batch data (Tensors are ALREADY on GPU)
            batch_data = patches_with_coords[i : i + BATCH_SIZE]

            batch_patches_list = [item[0] for item in batch_data]
            batch_coords_list = [item[1] for item in batch_data]

            batch_input = torch.cat(batch_patches_list, dim=0)

            # 2. Run inference
            output_batch = model(batch_input)

            # 3. Split, BUT KEEP ON GPU
            # We removed .cpu() here. The tensors stay in VRAM.
            split_output_patches = output_batch.split(1, dim=0)

            # 4. Store GPU tensors in the list
            for output_patch, coords in zip(split_output_patches, batch_coords_list):
                processed_patches_with_coords.append((output_patch, coords))

            # Optional: Reduce print frequency to save CPU/IO overhead
            #if (i + BATCH_SIZE) % (BATCH_SIZE * 5) == 0:
            if True:
                print(f"  Processed {i + len(batch_data)}/{num_patches_total} patches...")

    # --- 6. Stitch Patches Together (ON GPU) ---
    print("Stitching processed patches together on GPU...")

    # The stitch function will now run entirely on CUDA
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
    base64_image_string = "" # Initialize
    try:
        # Call the modified function which now returns a string
        base64_image_string = save_image(final_output_tensor)
    except Exception as e:
        print(f"❌ Error encoding image: {e}")

    # Return both the base64 string and the number of patches
    return base64_image_string, len(patches_with_coords)

def handler(job):
    
    # Capture the tuple (image_string, num_patches) returned by main
    image_string, num_patches = main(job["input"])
    
    return { "images": image_string,       # Place the base64 string here
             "num_patches": num_patches
           }

if __name__ == "__main__":
    print("🎯 Starting Deblur Handler")
    runpod.serverless.start({"handler": handler})
