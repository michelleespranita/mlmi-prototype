from keras.models import load_model

def get_trained_patient_CNN_model():
    model_CT = load_model("CT_Morbidity.model")
    
    return model_CT

def main():
    model_CT = get_trained_patient_CNN_model()


if __name__=="__main__":
    main()
