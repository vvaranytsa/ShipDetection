import numpy as np
from config import *
import os
from skimage.io import imread
import cv2


def rle_decode(mask_rle, shape=(768, 768)):
    encoded_pixels = np.array(mask_rle.split(), dtype=int)
    starts = encoded_pixels[::2] - 1
    ends = starts + encoded_pixels[1::2]
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def rle_code(img, shape=(768, 768)) -> str:
    img = img.astype('float32')
    img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    img = np.stack(np.vectorize(lambda x: 0 if x < 0.1 else 1)(img), axis=1)
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def masks_as_image(in_mask_list):
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks

def gen_pred(test_dir, img, model):
    rgb_path = os.path.join(test_dir, img)
    img = cv2.imread(rgb_path)
    img = img[::img_scaling[0], ::img_scaling[1]]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = np.squeeze(pred, axis=0)
    return cv2.imread(rgb_path), pred

def make_image_gen(in_df, batch_size=batch):
    image_ids = in_df['ImageId'].unique()  # Get unique image IDs
    num_images = len(image_ids)
    image_indices = np.arange(num_images)  # Create an array of indices for shuffling

    while True:
        np.random.shuffle(image_indices)  # Shuffle the indices

        for batch_start in range(0, num_images, batch_size):
            batch_indices = image_indices[batch_start : batch_start + batch_size]  # Select indices for the current batch
            batch_image_ids = image_ids[batch_indices]  # Get the image IDs for the current batch

            batch_rgb_images = []
            batch_masks = []

            for image_id in batch_image_ids:
                rgb_path = os.path.join(train_root, image_id)
                rgb_image = imread(rgb_path)

                mask_df = in_df[in_df['ImageId'] == image_id]
                mask_image = np.expand_dims(masks_as_image(mask_df['EncodedPixels'].values), -1)

                if img_scaling is not None:
                    rgb_image = rgb_image[::img_scaling[0], ::img_scaling[1]]
                    mask_image = mask_image[::img_scaling[0], ::img_scaling[1]]

                batch_rgb_images.append(rgb_image)
                batch_masks.append(mask_image)

            yield np.stack(batch_rgb_images, 0) / 255.0, np.stack(batch_masks, 0).astype(np.float32)

