import runpod


def handler(job):
    print(job["input"])
    return { "images":"[her are some base64 images]"
            }

if __name__ == "__main__":
    print("🎯 Starting Deblur Handler")
    runpod.serverless.start({"handler": handler})

