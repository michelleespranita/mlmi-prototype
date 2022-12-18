from collections import Counter
import os

import cv2 
import numpy as np
from skimage import measure


def split_target_dir(target_dir, output_dir):
    # get a List[List[str]] of paths to all the images that a single patient has
    patient_target_dirs = [os.path.join(target_dir, patient_dir, "CT") for patient_dir in os.listdir(target_dir)]
    image_files = [[image_file for image_file in os.listdir(patient_dir)] for patient_dir in patient_target_dirs]

    patient_output_dirs = [os.path.join(output_dir, patient_dir, "CT") for patient_dir in os.listdir(target_dir)]

    for patient_dir in patient_output_dirs:
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)

    for image_files_patient, patient_output_dir, patient_target_dir in zip(image_files, patient_output_dirs, patient_target_dirs):
        for image_file_patient in image_files_patient:
            image_path_target = os.path.join(patient_target_dir, image_file_patient)
            image_path_output = os.path.join(patient_output_dir, image_file_patient)

            img_split = split_lung_parenchyma(image_path_target, 15599, -96)
            cv2.imencode('.jpg', img_split)[1].tofile(image_path_output)

        print(f'Target list {patient_target_dir} done with {len(image_files_patient)} items.')


def split_lung_parenchyma(target, size, thr):
    img = cv2.imdecode(np.fromfile(target, dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
    try:
        img_thr = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, size, thr).astype(np.uint8)
    except:
        img_thr= cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 999, thr).astype(np.uint8)

    img_thr = 255 - img_thr
    img_test = measure.label(img_thr, connectivity=1)
    props = measure.regionprops(img_test)
    img_test.max()
    areas = [prop.area for prop in props]
    ind_max_area = np.argmax(areas) + 1
    del_array = np.zeros(img_test.max() + 1)
    del_array[ind_max_area] = 1
    del_mask = del_array[img_test]
    img_new = img_thr * del_mask
    mask_fill = fill_water(img_new)
    img_new[mask_fill == 1] = 255
    img_new = 255 - img_new
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(img_new.astype(np.uint8))
    labels = np.array(labels, dtype=float)
    maxnum = Counter(labels.flatten()).most_common(3)
    maxnum = sorted([x[0] for x in maxnum])
    background = np.zeros_like(labels)
    if len(maxnum) == 1:
        pass
    elif len(maxnum) == 2:
        background[labels == maxnum[1]] = 1
    else:
        background[labels == maxnum[1]] = 1
        background[labels == maxnum[2]] = 1
    img_new[background == 0] = 0
    img_new = cv2.dilate(img_new, np.ones((5,5),np.uint8) , iterations=3)
    img_new = cv2.erode(img_new, np.ones((5, 5), np.uint8), iterations=2)
    img_new = cv2.morphologyEx(img_new, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)), iterations=2)
    img_new = cv2.medianBlur(img_new.astype(np.uint8), 21)
    img_out = img * img_new.astype(bool)
    return img_out


def fill_water(img):
    copyimg = img.copy()
    copyimg.astype(np.float32)
    height, width = img.shape
    img_exp = np.zeros((height + 20,width + 20))
    height_exp, width_exp = img_exp.shape
    img_exp[10:-10, 10:-10] = copyimg
    mask1 = np.zeros([height + 22, width + 22],np.uint8)   
    mask2 = mask1.copy()
    mask3 = mask1.copy()
    mask4 = mask1.copy()
    cv2.floodFill(np.float32(img_exp), mask1, (0, 0), 1) 
    cv2.floodFill(np.float32(img_exp), mask2, (width_exp-1, height_exp-1), 1)
    cv2.floodFill(np.float32(img_exp), mask3, (width_exp-1, 0), 1)
    cv2.floodFill(np.float32(img_exp), mask4, (0, height_exp-1), 1)
    mask = mask1 | mask2 | mask3 | mask4
    output = mask[1:-1, 1:-1][10:-10, 10:-10]
    return output


target_dir = "sample_image_dataset"
output_dir = "sample_image_dataset_with_extracted_lung_parenchyma"
split_target_dir(target_dir, output_dir)
