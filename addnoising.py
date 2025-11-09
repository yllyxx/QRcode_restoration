import os
import random
import numpy as np
import cv2
import torch
import utils_images as util
from tqdm import tqdm

def add_noise_to_image(img, sigma_min=0, sigma_max=75):
    """
    Add noise to an image.
    Args:
        img (numpy array): Input image.
        sigma_min (float): Minimum noise level.
        sigma_max (float): Maximum noise level.
    Returns:
        noisy_img (numpy array): Noisy image.
        noise_level_map (torch tensor): Noise level map.
    """
    img_H = util.uint2tensor3(img)
    img_L = img_H.clone()

    noise_level = torch.FloatTensor([np.random.uniform(sigma_min, sigma_max)])/255.0
    noise_level_map = torch.ones((1, img_L.size(1), img_L.size(2))).mul_(noise_level).float()
    noise = torch.randn(img_L.size()).mul_(noise_level).float()
    img_L.add_(noise)

    img_L = torch.cat((img_L, noise_level_map), 0)
    noisy_img = util.tensor2uint(img_L[:3, :, :])

    return noisy_img, noise_level_map

def process_images(input_folder, output_folder, sigma_min=0, sigma_max=75):
    """
    Process all images in the input folder, add noise and save to the output folder.
    Args:
        input_folder (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        sigma_min (float): Minimum noise level.
        sigma_max (float): Maximum noise level.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = util.get_image_paths(input_folder)
    for image_path in tqdm(image_paths, desc='Processing images'):
        img = util.imread_uint(image_path, n_channels=3)
        noisy_img, _ = add_noise_to_image(img, sigma_min, sigma_max)
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        util.imsave(noisy_img, output_path)

if __name__ == '__main__':
    input_folder = r'D:\project\KAIR-master\trainsets\trainH'
    output_folder = 'waterloo_train_noise'
    sigma_min = 36.057
    sigma_max = 80.631

    process_images(input_folder, output_folder, sigma_min, sigma_max)
