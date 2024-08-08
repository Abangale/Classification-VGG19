# Import necessary libraries and modules
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models, transforms
import time
from typing import List

MODEL_DIRECTORY = './LungNet22.pth' # Model Location
use_gpu = torch.cuda.is_available()

def convert_byte_to_arr(byte_image):
    """
    Convert an image in byte format to a PIL Image object (RGB format).

    Args:
        byte_image (bytes): The image data in byte format.

    Returns:
        Image.Image: A PIL Image object representing the image in RGB format.
    """
    arr_image = Image.open(BytesIO(byte_image)).convert("RGB")
    return arr_image


def convert_arr_to_byte(arr_image):
    """
    Convert a numpy array image (RGB format) to byte format (JPEG).

    Args:
        arr_image (numpy.ndarray): The image data as a numpy array in RGB format.

    Returns:
        bytes: The image data in byte format (JPEG).
    """
    arr_image = np.array(arr_image)
    arr_image = cv2.cvtColor(arr_image, cv2.COLOR_RGB2BGR)
    # Encode the image as JPEG format
    success, byte_image = cv2.imencode(".jpg", arr_image)
    if success:
        return byte_image.tobytes()
    else:
        raise Exception("Cannot convert array image to byte image")
    

def multiple_to_one(images):
    """
    Combine multiple images horizontally into a single image.

    Args:
        images (List[Image.Image]): List of PIL Image objects representing the input images.

    Returns:
        Image.Image: A new PIL Image object containing the input images combined horizontally.
    """
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    return new_im


def assign_image_label(images: List[Image.Image], labels: List[str], confs: List[float], font="arial.ttf", font_size=25) -> List[Image.Image]:
    """
    Add labels to the input images.

    Args:
        images (List[Image.Image]): List of PIL Image objects representing the input images.
        labels (List[str]): List of labels corresponding to the input images.
        confs (List[float]): List of confidence level of each prediction for the corresponding input image.
        font (str, optional): The font file to be used for the labels. Defaults to "arial.ttf".
        font_size (int, optional): The font size for the labels. Defaults to 25.

    Returns:
        List[Image.Image]: List of PIL Image objects with labels added to the top left corner.
    """
    image_w_label = []
    font_setting = ImageFont.truetype(font, font_size)

    # Debug information
    print(f"Total images: {len(images)}, labels: {len(labels)}, confs: {len(confs)}")

    for index in range(len(images)):
        if index < len(labels) and index < len(confs):
            label_text = f"{labels[index]} ({confs[index]:.4f})"
        else:
            label_text = "Unknown (0.0000)"
            print(f"[WARNING] No label/conf for image index {index}")

        I1 = ImageDraw.Draw(images[index])
        I1.text((10, 10), label_text, fill=(255, 255, 0), font=font_setting)
        image_w_label.append(images[index])
    
    return image_w_label
    

def get_data(np_images):
    """
    Prepare the list of numpy array images for classification.

    Args:
        np_images (List[numpy.ndarray]): List of numpy array images (RGB format).

    Returns:
        List[torch.Tensor]: List of preprocessed images as PyTorch tensors.
    """
    data_transform = transforms.Compose([
        transforms.Resize(254),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    data = []
    for image in np_images:
        # Convert numpy ndarray [3, 224, 224] to PyTorch tensor
        image = data_transform(image)
        # Expand to [batch_size, 3, 224, 224]
        image = torch.unsqueeze(image, 0)
        data.append(image)
    return data


def get_vgg19_pretrained_model(model_dir=MODEL_DIRECTORY, weights=models.VGG19_Weights.DEFAULT):
    """
    Retrieve the VGG-19 pre-trained model and modify the classifier with a fine-tuned one.

    Args:
        model_dir (str): Directory path for loading a pre-trained model state dictionary.
        weights (str or dict): Pre-trained model weights.

    Returns:
        torchvision.models.vgg19: VGG-19 model with modified classifier.
    """
    print("[INFO] Getting VGG-19 pre-trained model...")
    vgg19 = models.vgg19(weights=weights)
    # Freeze training for all layers
    for param in vgg19.features.parameters():
        param.requires_grad = False
    # Get number of features in the last layer
    num_features = vgg19.classifier[-1].in_features
    # Remove the last layer
    features = list(vgg19.classifier.children())[:-1]
    # Add custom layer with custom number of output classes (10 for your case)
    features.extend([nn.Linear(num_features, 10)])
    # Replace the model's classifier
    vgg19.classifier = nn.Sequential(*features)
    # Load the pretrained model
    vgg19.load_state_dict(torch.load(model_dir), strict=False)
    vgg19.eval()
    print("[INFO] Loaded VGG-19 pre-trained model\n", vgg19, "\n")

    return vgg19


def get_prediction(model, images):
    """
    Perform image classification using the provided model.

    Args:
        model (torchvision.models.vgg16_bn): The fine-tuned VGG-16 model.
        images (List[torch.Tensor]): List of preprocessed images as PyTorch tensors.

    Returns:
        Tuple[List[str], List[float], float]: A tuple containing the list of predicted labels, the confidence for the predictions, and the time taken for classification.
    """
    since = time.time()
    labels = []
    confs = []
    model.train(False)
    model.eval()

    for image in images:
        with torch.no_grad():
            if use_gpu:
                image = Variable(image.cuda())
            else:
                image = Variable(image)

        outputs = model(image)
        
        probs = torch.nn.functional.softmax(outputs.data, dim=1)
        conf, pred = torch.max(probs, 1)
        
        if pred == 0:
            labels.append('control')
            confs.append(round(float(conf.cpu()), 4))
        elif pred == 1:
            labels.append('covid-19')
            confs.append(round(float(conf.cpu()), 4))
        elif pred == 2:
            labels.append('effusion')
            confs.append(round(float(conf.cpu()), 4))
        elif pred == 3:
            labels.append('lung_capacity')
            confs.append(round(float(conf.cpu()), 4))
        elif pred == 4:
            labels.append('mass')
            confs.append(round(float(conf.cpu()), 4))
        elif pred == 5:
            labels.append('nodule')
            confs.append(round(float(conf.cpu()), 4))
        elif pred == 6:
            labels.append('pulmonary_fibriosis')
            confs.append(round(float(conf.cpu()), 4))
        elif pred == 7:
            labels.append('pneumonia')
            confs.append(round(float(conf.cpu()), 4))
        elif pred == 8:
            labels.append('pneumothorax')
            confs.append(round(float(conf.cpu()), 4))
        elif pred == 9:
            labels.append('tuberculosis')
            confs.append(round(float(conf.cpu()), 4))
        else:
            print('[INFO] Labeling went wrong')

    elapsed_time = time.time() - since

    return labels, confs, elapsed_time
