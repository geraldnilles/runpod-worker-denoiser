import runpod
import os
import base64
import sys

# --- Helper Function ---
def encode_image_to_base64(filepath: str) -> str:
    """Reads an image file and returns it as a base64 encoded string."""
    try:
        with open(filepath, "rb") as image_file:
            img_bytes = image_file.read()
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            return base64_string
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at: {filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading or encoding file: {e}", file=sys.stderr)
        sys.exit(1)

# --- Argument Check ---
if len(sys.argv) < 2:
    print("Usage: python client.py <path_to_image.jpg>", file=sys.stderr)
    sys.exit(1)

image_filepath = sys.argv[1]

# --- Configuration ---
# Load API key and Endpoint ID from environment variables
api_key = os.getenv("RUNPOD_API_KEY")
endpoint_id = os.getenv("ENDPOINT_ID")

if not api_key:
    print("Error: RUNPOD_API_KEY environment variable not set.", file=sys.stderr)
    sys.exit(1)

if not endpoint_id:
    print("Error: ENDPOINT_ID environment variable not set.", file=sys.stderr)
    sys.exit(1)

runpod.api_key = api_key
endpoint = runpod.Endpoint(endpoint_id)

# --- Prepare Payload ---
print(f"Encoding image: {image_filepath}...")
base64_image_data = encode_image_to_base64(image_filepath)

# --- Run Job ---
print(f"Sending request to endpoint: {endpoint_id}...")

try:
    # Send the request with the base64 image data.
    run_request = endpoint.run(
        {
            "input": {
                "image": base64_image_data  # <-- Modified to use the encoded string
            }
        }
    )

    # Check initial status
    status = run_request.status()
    print(f"Initial job status: {status}")

    if status != "COMPLETED":
        # Poll for results with timeout
        output = run_request.output(timeout=120) 
    else:
        output = run_request.output()
    
    # --- Process Response ---
    if output and "images" in output and "num_patches" in output:
        base64_image_string = output["images"]
        num_patches = output["num_patches"]
        
        print(f"Job processed {num_patches} patches.")

        if base64_image_string:
            print("Decoding base64 image...")
            try:
                # Decode the base64 string into bytes
                image_bytes = base64.b64decode(base64_image_string)

                # Write the bytes to a file
                output_filename = "output.jpg"
                with open(output_filename, "wb") as f:
                    f.write(image_bytes)

                print(f"✅ Successfully saved image to: {output_filename}")

            except base64.binascii.Error as e:
                print(f"❌ Error: Failed to decode base64 string. {e}", file=sys.stderr)
            except Exception as e:
                print(f"❌ Error writing file: {e}", file=sys.stderr)
        else:
            print("❌ Error: Received an empty 'images' field in the response.", file=sys.stderr)

    else:
        print(f"❌ Error: Unexpected response format from endpoint.", file=sys.stderr)
        print(f"Received: {output}", file=sys.stderr)

except TimeoutError:
    print("❌ Error: The job timed out (client-side timeout).", file=sys.stderr)
