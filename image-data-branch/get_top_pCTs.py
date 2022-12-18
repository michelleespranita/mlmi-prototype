import csv
import os

import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from tqdm import tqdm


def create_csv_with_top_pCTs(CT_images_model, dict_from_patient_target_dir_to_image_files, top_k=10, invalid_cutoff=0.5, time_seq=True):
    # each csv row will have 2 cells:
    # 1. path to the patient_target_dir as a string (e.g. "sample_image_dataset_with_extracted_lung_parenchyma/Patient 6/CT")
    # 2. file names of the top_k positive CTs as a string (e.g. "IMG-0001-00028.jpg, IMG-0001-00029.jpg, IMG-0001-00038.jpg")
    csv_rows = []

    for patient_target_dir, image_files in tqdm(dict_from_patient_target_dir_to_image_files.items()):
        X_CT_Valid = np.array([read_ct_img_bydir(os.path.join(patient_target_dir, image_file)) for image_file in image_files])
        y_CT_Valid = CT_images_model.predict(X_CT_Valid)

        probs_df = pd.DataFrame({"images": image_files, 'NiCT':y_CT_Valid[:,0], 'pCT':y_CT_Valid[:,1], 'nCT':y_CT_Valid[:,2]})

        # remove all non-informative CTs
        probs_df = probs_df[probs_df['NiCT'] <= invalid_cutoff]

        probs_df.sort_values('pCT', ascending=0, inplace=True)

        # get the top 10 pCTs for every patient
        if time_seq:
            probs_df = probs_df.head(top_k).sort_index()
        else:
            probs_df = probs_df.head(top_k)
        if len(probs_df) < top_k:
            print(f'{patient_target_dir} does not have enough CTs.')

        top_k_images = probs_df["images"].tolist()

        # convert list to a single string
        top_k_images = ", ".join(top_k_images)

        # save them in the csv_rows list together with patient_target_dir
        csv_rows.append([patient_target_dir, top_k_images])

    # save the information in a csv file together with a header
    header = ["patient_directory", "top_k_image_files"]

    with open("top_pCTs.csv", "w") as fp:
        csv_writer = csv.writer(fp)

        csv_writer.writerow(header)
        csv_writer.writerows(csv_rows)


def read_ct_img_bydir(target_dir):
    img = cv2.imdecode(np.fromfile(target_dir, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 200))
    return img


def get_images_paths():
    target_dir = "sample_image_dataset_with_extracted_lung_parenchyma"
    patient_target_dirs = [os.path.join(target_dir, patient_dir, "CT") for patient_dir in os.listdir(target_dir)]
    image_files = [[image_file for image_file in os.listdir(patient_dir)] for patient_dir in patient_target_dirs]

    dict_from_patient_target_dir_to_image_files = {patient_dir: image_files_patient for patient_dir, image_files_patient in zip(patient_target_dirs, image_files)}

    return dict_from_patient_target_dir_to_image_files


def get_trained_CT_images_model():
    model = load_model("CT_images.model")
    return model


def main():
    CT_images_model = get_trained_CT_images_model()
    dict_from_patient_target_dir_to_image_files = get_images_paths()

    create_csv_with_top_pCTs(CT_images_model, dict_from_patient_target_dir_to_image_files)

if __name__=="__main__":
    main()
