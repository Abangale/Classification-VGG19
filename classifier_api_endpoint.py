from fastapi import FastAPI, UploadFile, Response
from fastapi.responses import HTMLResponse
from PIL import Image
import torch
from classifier_function import convert_byte_to_arr
from classifier_function import convert_arr_to_byte
from classifier_function import multiple_to_one
from classifier_function import assign_image_label
from classifier_function import get_data
from classifier_function import get_vgg19_pretrained_model
from classifier_function import get_prediction

FONT = "arialbd.ttf" # Font Family
FONT_SIZE = 76 # Label Size 
NEW_WIDTH = 400 # Output Image Size
use_gpu = torch.cuda.is_available()

app = FastAPI()
@app.get("/")
def welcome_page():
    """
    Serves the root route ("/") and displays a welcome message with a link to the API documentation.

    Returns:
        fastapi.responses.HTMLResponse: HTML response with a welcome message and a link to the API documentation.
    """
    return HTMLResponse(
        """
        <h1>Welcome to Banana</h1>
        <p>Click the button below to go to /docs/:</p>
        <form action="/docs" method="get">
            <button type="submit">Visit Website</button>
        </form>
    """
    )


@app.post("/classification_predict")
async def classification_predict(in_images: list[UploadFile]):
    """
    API endpoint to classify multiple images using a fine-tuned VGG-19 model.

    Args:
        in_images (List[UploadFile]): List of images in JPG format to be classified.

    Returns:
        fastapi.responses.Response: Images with a label on the top left corner as a response.
    """
    images = []
    for in_image in in_images:
        byte_image = await in_image.read()
        arr_image = convert_byte_to_arr(byte_image)
        images.append(arr_image)
    print(f'[INFO] Received {len(images)} images')
    
    # Preparing data and loading the model
    data = get_data(images)
    vgg = get_vgg19_pretrained_model()
    
    # Use GPU if available
    print('[INFO] Classification in progress')
    print("[INFO] Using CUDA") if use_gpu else print("[INFO] Using CPU")
    if use_gpu:
        torch.cuda.empty_cache()
        vgg.cuda()
    
    labels, confs, elapsed_time = get_prediction(vgg, data)
    print(f"[INFO] Label : {labels} with confidence {confs} in time {(elapsed_time // 60):.0f}m {(elapsed_time % 60):.0f}s")

    # Add label and confidence level to the top left corner of the input image
    print('[INFO] Writing labels onto output images')
    image_w_label = assign_image_label(images, labels, confs, font=FONT, font_size=FONT_SIZE)
    
    # Combine multiple images into one
    image_combined = multiple_to_one(image_w_label)
    
    # Resize the combined image to a lower resolution (e.g., width = 800)
    new_width = NEW_WIDTH
    aspect_ratio = new_width / float(image_combined.size[0])
    new_height = int((float(image_combined.size[1]) * float(aspect_ratio)))
    image_resized = image_combined.resize((new_width, new_height), Image.LANCZOS)
    
    # Output API
    print('[INFO] Returning output')
    byte_images = convert_arr_to_byte(image_resized)
    response_text = f'Label : {labels} with confidence {confs} in time {(elapsed_time // 60):.0f}m {(elapsed_time % 60):.0f}s'
    response = Response(content=byte_images, media_type="image/jpg")
    response.headers["Result"] = response_text
    
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
