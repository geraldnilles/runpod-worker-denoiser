import runpod
import os
import base64
import sys

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

# --- Run Job ---
print(f"Sending request to endpoint: {endpoint_id}...")

try:
    # Send the request. The input payload doesn't matter for your
    # current handler, but we send a valid dict.
    # Increased timeout to 600s for image processing.
    output = endpoint.run_sync(
        {
            "input": {
                "message": "Starting denoising job..."
            }
        },
        timeout=120,  # 2 minute client timeout
    )

    print("Request completed.")
   
    #print(run_request)
    output
    # --- Process Response ---
    if "images" in output and "num_patches" in output:
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
