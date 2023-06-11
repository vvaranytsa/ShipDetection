import tensorflow as tf
from config import *
from model_funcs.func import generate_prediction
from model_funcs.unet import IoU_loss, dice_coefficient
from matplotlib import pyplot as plt
import os
import random

# Load the model
object_scope = {'IoU_loss': IoU_loss, 'dice_coefficient': dice_coefficient}
final_model = tf.keras.models.load_model('model_folder/final_model.h5', custom_objects=object_scope)
test_files = os.listdir(test_root)

# Choose a random subset of images to check
num_images_to_check = 5
check = random.sample(test_files, num_images_to_check)

# Loop over the images
for image_file in check:
    # Make predictions
    img, pred = generate_prediction(test_root, image_file, final_model)
    # Plot the images
    fig, axs = plt.subplots(1, 2, figsize=(5, 5))
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title("Image")
    axs[1].imshow(pred, interpolation='bilinear')
    axs[1].axis('off')
    axs[1].set_title("Prediction")
    plt.show()