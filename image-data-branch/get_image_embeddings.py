from keras.models import load_model

def get_trained_CT_Morbidity_model():
    model = load_model("CT_Morbidity.model")
    return model

def main():
    CT_Morbidity_model = get_trained_CT_Morbidity_model()


if __name__=="__main__":
    main()
