from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import json
from html import unescape
app = FastAPI()

raw_json = """
[
    {"name": "5x_1.png", "link": "https://example.com/images/5x_1.png"},
    {"name": "5x_2.png", "link": "https://example.com/images/5x_2.png"},
    {"name": "5x_3.png", "link": "https://cad.onshape.com/documents/78d69674ed186dd2164efe20/w/14520a522f98a7211ef0f35e/e/f378613f69584d063ca6593e?renderMode=0&uiState=657b05059e258f3ce790710b"},
    {"name": "5x_5.png", "link": "https://example.com/images/5x_5.png"},
    {"name": "another_test.jpeg", "link": "https://example.com/images/another_test.jpeg"},
    {"name": "Screenshot 2023-11-08 at 1.06.38 PM.png", "link": "https://cad.onshape.com/documents/78d69674ed186dd2164efe20/w/14520a522f98a7211ef0f35e/e/8a2c7eb6e95098c309fd2f59?renderMode=0&uiState=657afde69e258f3ce78f6daf"},
    {"name": "Screenshot 2023-11-08 at 1.07.28 PM.png", "link": "https://cad.onshape.com/documents/78d69674ed186dd2164efe20/w/14520a522f98a7211ef0f35e/e/ab62e8c39b68ff91652a21cd?renderMode=0&uiState=657afe0f9e258f3ce78f6e15"},
    {"name": "Screenshot 2023-11-08 at 1.08.20 PM.png", "link": "https://cad.onshape.com/documents/78d69674ed186dd2164efe20/w/14520a522f98a7211ef0f35e/e/870ab50d1e7635126cf6c8c5?renderMode=0&uiState=657afe309e258f3ce78f6eb2"},
    {"name": "Screenshot 2023-11-08 at 1.08.41 PM.png", "link": "https://cad.onshape.com/documents/78d69674ed186dd2164efe20/w/14520a522f98a7211ef0f35e/e/c107c73089382bdedabf07a4?renderMode=0&uiState=657b01959e258f3ce790568e"},
    {"name": "ss_2.png", "link": "https://cad.onshape.com/documents/78d69674ed186dd2164efe20/w/14520a522f98a7211ef0f35e/e/e35fb2a39f8f24fddd720749?renderMode=0&uiState=657afda09e258f3ce78f6b49"}
]
"""

# Load the JSON string into a Python object
json_data = json.loads(raw_json)


# Create "static" directory if it doesn't exist
static_directory = "static/"
if not os.path.exists(static_directory):
    os.makedirs(static_directory)

app.mount("/static", StaticFiles(directory=static_directory), name="static")
templates = Jinja2Templates(directory="templates")

# Define the model and related components
model = models.resnet18(pretrained=True)
model.eval()

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


model.avgpool.register_forward_hook(get_activation("avgpool"))

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to read data
def read_data():
    all_vecs = np.load("all_vecs.npy", allow_pickle=True)
    all_names = np.load("all_names.npy", allow_pickle=True)
    return all_vecs, all_names

vecs, names = read_data()

def find_data(json_data, name_to_find):
    # Decode HTML entities in the name_to_find
    name_to_find_decoded = unescape(name_to_find)

    name_to_find_decoded = name_to_find_decoded.replace('\u00A0', ' ')
    name_to_find_decoded = name_to_find_decoded.replace('&nbsp;', ' ')
    name_to_find_decoded = name_to_find_decoded.replace('\u00A0', ' ').replace('&nbsp;', ' ').replace('\u2009', ' ').replace('\u202F', ' ')


    # Search for the specified name in the JSON data
    found_data = next(
        (item for item in json_data if item["name"] == name_to_find or item["name"] == name_to_find_decoded),
        None
    )

    return found_data

@app.post("/upload")
async def upload(image: UploadFile = File(...)):
    threshold = 7.5
    try:
        # Save the received image temporarily
        file_path = f"{static_directory}{image.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(image.file.read())

        # Calculate the vector for the provided image
        img = Image.open(file_path).convert('RGB')
        img = transform(img)
        with torch.no_grad():
            out = model(img[None, ...])
            target_vec = activation["avgpool"].cpu().numpy().squeeze()[None, ...]

        # Calculate cosine similarity between the new image and all images in the dataset
        similarities = cosine_similarity(target_vec.reshape(1, -1), vecs).flatten()

        # Get the indices of top similar images
        top_similar_indices = similarities.argsort()[::-1]

        scaled_similarities = [1 + (similarity * 9) for similarity in similarities]

        # base_url = "http://127.0.0.1:9000/static/"
        base_url = "https://fastapi-example-40q5.onrender.com/static/"




        # Prepare JSON response data
        json_response_data = {

            "data": [
                {
                    "index": i + 1,
                    "name": names[index],
                    "test_url": find_data(json_data,names[index])['link'],
                    "similarity": round(scaled_similarities[index], 2) *  10,  # Round to two decimal places
                    "image_url": f"{base_url}{names[index]}"  # Complete image URL
                }
                for i, index in enumerate(top_similar_indices)

                if scaled_similarities[index] >= threshold
            ],

        }
        return json_response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/advanceSearch")
async def advanceSearch(image: UploadFile = File(...)):
    try:
        # Save the received image temporarily
        file_path = f"{static_directory}{image.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(image.file.read())

        # Calculate the vector for the provided image
        img = Image.open(file_path).convert('RGB')
        img = transform(img)
        with torch.no_grad():
            out = model(img[None, ...])
            target_vec = activation["avgpool"].cpu().numpy().squeeze()[None, ...]

        # Calculate cosine similarity between the new image and all images in the dataset
        similarities = cosine_similarity(target_vec.reshape(1, -1), vecs).flatten()

        # Get the indices of top similar images
        top_similar_indices = similarities.argsort()[::-1]

        scaled_similarities = [1 + (similarity * 9) for similarity in similarities]

        # base_url = "http://127.0.0.1:9000/static/"
        base_url = "https://fastapi-example-40q5.onrender.com/static/"

        # Prepare JSON response data
        json_response_data = {
            "data": [
                {
                    "index": i + 1,
                    "name": names[index],
                    "similarity": round(scaled_similarities[index], 2) *  10,  # Round to two decimal places
                    "image_url": f"{base_url}{names[index]}"  # Complete image URL
                }
                for i, index in enumerate(top_similar_indices)
            ],
        }
        return json_response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9000)
