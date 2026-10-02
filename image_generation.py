import openai
import base64

# Don't forget to fill out your API k
client = openai.OpenAI(api_key="<YOUR ETH-SPH API KEY HERE>",base_url="https://litellm.sph-prod.ethz.ch/v1") 

# Specify what you want on your image and the model you want to use
prompt = "A beautiful ankle boot inspired by flowers for the new spring summer collection"
model = "azure/dall-e-3"

# Sending a request to the model
result = client.images.generate(
    model=model,
    prompt=prompt,
    response_format="b64_json" # The result will be returned in bytecode
)

# Converting the received bytecode to something python can save as an image
image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Saving the image
with open("image.jpg", "wb") as f:
    f.write(image_bytes)