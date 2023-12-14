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

app = FastAPI()


# app.mount("/train_images", StaticFiles(directory="../train_images"), name="train_images")


app.mount("/static", StaticFiles(directory="images"), name="static")
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


@app.post("/upload")
async def upload(image: UploadFile = File(...)):
    threshold = 8
    try:
        # Save the received image temporarily
        file_path = f"static/{image.filename}"
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

        base_url = "https://fastapi-example-40q5.onrender.com/static/"

        # Prepare JSON response data
        json_response_data = {
            # "provided_image": image.filename,
            "data": [
                {
                    "index": i + 1,
                    "name": names[index],
                    "similarity": round(scaled_similarities[index], 2) *  10,  # Round to two decimal places
                    "image_url": f"{base_url}{names[index]}"  # Complete image URL
                }
                for i, index in enumerate(top_similar_indices)
                if scaled_similarities[index] >= threshold
            ],
        }
        return json_response_data


        # return {"top_similar_images": [
        #             {"index": i + 1, "name": names[index], "similarity": similarities[index]}
        #             for i, index in enumerate(top_similar_indices)
        #         ]
        # }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/advanceSearch")
async def advanceSearch(image: UploadFile = File(...)):
    try:
        # Save the received image temporarily
        file_path = f"static/{image.filename}"
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

        base_url = "https://fastapi-example-40q5.onrender.com/static/"

        # Prepare JSON response data
        json_response_data = {
            # "provided_image": image.filename,
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


        # return {"top_similar_images": [
        #             {"index": i + 1, "name": names[index], "similarity": similarities[index]}
        #             for i, index in enumerate(top_similar_indices)
        #         ]
        # }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9000)
