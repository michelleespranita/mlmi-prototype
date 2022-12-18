import csv
import os

import numpy as np
from keras.models import load_model, Model
from tqdm import tqdm

from get_top_pCTs import read_ct_img_bydir


def create_image_embeddings(CT_Morbidity_model_without_linear_layers, dict_from_patient_target_dir_to_top_pCTs):
    # list to store the image_embeddings
    image_embeddings = []

    for patient_target_dir, top_pCTs in tqdm(dict_from_patient_target_dir_to_top_pCTs.items()):
        # top_pCTs is of shape (10, 200, 200)
        top_pCTs = np.array([read_ct_img_bydir(os.path.join(patient_target_dir, image_file)) for image_file in top_pCTs])

        # but model expect (batch_size, 200, 200, 10) as input
        # so add a new axis and tranpose accordingly
        top_pCTs = top_pCTs[:,:,:,np.newaxis].transpose(3,1,2,0)

        # top_pCT_embedding of shape (1, 10000)
        top_pCT_embedding = CT_Morbidity_model_without_linear_layers.predict(top_pCTs)

        # store the embedding in the list
        image_embeddings.append(top_pCT_embedding[0])

    # image_embeddings of shape (num_patients, 10000)
    image_embeddings = np.stack(image_embeddings)

def get_top_pCTs_paths():
    dict_from_patient_target_dir_to_top_pCTs = {}

    with open("top_pCTS.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # skip the first line (i.e. the header line)
        next(csv_reader)

        for row in csv_reader:
            patient_directory = row[0]
            top_pCTs = row[1]

            # convert string to list
            top_pCTs = top_pCTs.split(", ")

            dict_from_patient_target_dir_to_top_pCTs[patient_directory] = top_pCTs

    return dict_from_patient_target_dir_to_top_pCTs


def remove_linear_layers_from_trained_model(CT_Morbidity_model):
    # the -6. layer is a Flatten layer that flattens the feature maps of size 25x25x16 to 10000
    # since we want the image embeddings, we specify the output of the new model to be after the Flatten layer
    # (i.e. before the dense/linear layers of the classification head)
    model = Model(inputs=CT_Morbidity_model.input, outputs=CT_Morbidity_model.layers[-6].output)
    return model

def get_trained_CT_Morbidity_model():
    model = load_model("CT_Morbidity.model")
    return model

def main():
    CT_Morbidity_model = get_trained_CT_Morbidity_model()
    CT_Morbidity_model_without_linear_layers = remove_linear_layers_from_trained_model(CT_Morbidity_model)

    dict_from_patient_target_dir_to_top_pCTs = get_top_pCTs_paths()

    create_image_embeddings(CT_Morbidity_model_without_linear_layers, dict_from_patient_target_dir_to_top_pCTs)


if __name__=="__main__":
    main()
