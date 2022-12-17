"""
WORK IN PROGRESS!
"""

import os

import cv2
import numpy as np
from keras.models import load_model


def read_ct_img_bydir(target_dir):
    img = cv2.imdecode(np.fromfile(target_dir, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 200))
    return img


def get_images_paths():
    target_dir = "sample_image_dataset_with_extracted_lung_parenchyma"
    patient_target_dirs = [os.path.join(target_dir, patient_dir, "CT") for patient_dir in os.listdir(target_dir)]
    image_files = [[image_file for image_file in os.listdir(patient_dir)] for patient_dir in patient_target_dirs]

    return patient_target_dirs, image_files


def load_CT_images_model():
    model = load_model("CT_images.model")
    return model


def main():
    print()
    # CT_images_model = load_CT_images_model()
    # patient_target_dirs, image_files = get_images_paths()
    # print(patient_target_dirs[0])