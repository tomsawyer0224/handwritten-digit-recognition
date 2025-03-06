import chainlit as cl
from PIL import Image, ImageOps
import numpy as np

@cl.on_message
async def main(message: cl.Message):
    images = [file for file in message.elements if "image" in file.mime]
    image_paths = [image.path for image in images]
    gray_images = [ImageOps.grayscale(Image.open(im_path)) for im_path in image_paths]
    resized_images = [gr_im.resize((28,28)) for gr_im in gray_images]
    normalized_images = [np.array(rs_im)/255.0 for rs_im in resized_images]
    await cl.Message(
        content=f"Predict: your image {np.array2string(normalized_images[0])}",
        #elements = images
    ).send()