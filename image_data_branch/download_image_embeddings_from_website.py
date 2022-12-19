import os
import csv
import zipfile
from tqdm import tqdm
import time
import webbrowser
import io
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model, Model
import pandas as pd
import shutil


def store_image_embeddings(image_embeddings, CT_Morbidity_model_without_linear_layers, patient_folder_path, top_pCTs):
    # top_pCTs is of shape (10, 200, 200)
    top_pCTs = np.array([read_ct_img_bydir(os.path.join(patient_folder_path, image_file)) for image_file in top_pCTs])

    # but model expect (batch_size, 200, 200, 10) as input
    # so add a new axis and tranpose accordingly
    top_pCTs = top_pCTs[:,:,:,np.newaxis].transpose(3,1,2,0)

    # top_pCT_embedding of shape (1, 10000)
    top_pCT_embedding = CT_Morbidity_model_without_linear_layers.predict(top_pCTs)

    # store the embedding in the list
    image_embeddings.append(top_pCT_embedding[0])


def get_top_pCTS(CT_images_model, images, image_files, top_k=10, invalid_cutoff=0.5, time_seq=True):
    probs = CT_images_model.predict(images)

    probs_df = pd.DataFrame({"images": image_files, 'NiCT':probs[:,0], 'pCT':probs[:,1], 'nCT':probs[:,2]})

    # remove all non-informative CTs
    probs_df = probs_df[probs_df['NiCT'] <= invalid_cutoff]

    probs_df.sort_values('pCT', ascending=0, inplace=True)

    # get the top 10 pCTs for every patient
    if time_seq:
        probs_df = probs_df.head(top_k).sort_index()
    else:
        probs_df = probs_df.head(top_k)
    if len(probs_df) < top_k:
        # patient does not have enough CTs
        return None

    top_k_images = probs_df["images"].tolist()
    return top_k_images


def read_ct_img_bydir(target_dir):
    img = cv2.imdecode(np.fromfile(target_dir, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 200))
    return img

def download_image_embeddings(CT_Morbidity_model_without_linear_layers, CT_images_model, patients):
    url_template = "https://ngdc.cncb.ac.cn/ictcf/patient/CT/Patient%20{}.zip"
    download_path = "/Users/michelleespranita/Downloads/"

    patient_failed_to_download = []

    # list to store the image_embeddings
    image_embeddings = []

    for patient_number in tqdm(patients):
        url = url_template.format(patient_number)

        try:
            webbrowser.open(url)
        except:
            patient_failed_to_download.append(patient_number)
            continue

        zip_file_path = os.path.join(download_path, f"Patient {patient_number}.zip")

        # wait until the zip file has been downloaded
        while not os.path.exists(zip_file_path):
            time.sleep(100)

        # extract the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)

        # delete the zip file
        os.remove(zip_file_path)

        patient_folder_path = os.path.join(download_path, f"Patient {patient_number}", "CT")
        image_files = [file for file in os.listdir(patient_folder_path)]
        images = np.array([read_ct_img_bydir(os.path.join(patient_folder_path, image_file)) for image_file in image_files])
        top_pCTs = get_top_pCTS(CT_images_model, images, image_files)

        # patient does not have enough CTs
        if top_pCTs is None:
            patient_failed_to_download.append(patient_number)
            continue

        # store the image embeddings for current patient
        store_image_embeddings(image_embeddings, CT_Morbidity_model_without_linear_layers, patient_folder_path, top_pCTs)

        # delete patient folder
        patient_folder_path = os.path.join(download_path, f"Patient {patient_number}")
        shutil.rmtree(patient_folder_path)
        
    # image_embeddings of shape (num_patients, 10000)
    image_embeddings = np.stack(image_embeddings)

    print(f"Image embedding size: {image_embeddings.size}")
    np.save("image_embeddings_full_dataset", image_embeddings)

    print(f"Failed to download patients: {patient_failed_to_download}")

def remove_linear_layers_from_trained_model(CT_Morbidity_model):
    # the -6. layer is a Flatten layer that flattens the feature maps of size 25x25x16 to 10000
    # since we want the image embeddings, we specify the output of the new model to be after the Flatten layer
    # (i.e. before the dense/linear layers of the classification head)
    model = Model(inputs=CT_Morbidity_model.input, outputs=CT_Morbidity_model.layers[-6].output)
    return model

def get_trained_CT_Morbidity_model():
    model = load_model("CT_Morbidity.model")
    return model


def get_trained_CT_images_model():
    model = load_model("CT_images.model")
    return model


def get_patients():
    patient_numbers = []
    patient_str = "Patient "

    with open("matt_metadata_norm_morbidity.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="\t")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        for row in csv_reader:
            patient = row[0]  # e.g. "Patient 123"
            patient_number = patient[len(patient_str):]  # e.g. "123"

            patient_numbers.append(patient_number)

    return patient_numbers


def main():
    patient_numbers = get_patients()

    CT_images_model = get_trained_CT_images_model()
    CT_Morbidity_model = get_trained_CT_Morbidity_model()
    CT_Morbidity_model_without_linear_layers = remove_linear_layers_from_trained_model(CT_Morbidity_model)

    download_image_embeddings(CT_Morbidity_model_without_linear_layers, CT_images_model, patient_numbers)

if __name__=="__main__":
    main()