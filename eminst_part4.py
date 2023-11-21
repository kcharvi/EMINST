import streamlit as st
from PIL import Image
import torch
from helper import VGG11, process_image, prediction_result

def main():
    model=VGG11()
    model.load_state_dict(
        torch.load('charviku_dmandava_assignment2_part4.h5',
                    map_location=torch.device('cpu')))
    model.eval()

    st.set_option("deprecation.showfileUploaderEncoding",
                   False)
    st.title("EMINST Classifier")
    st.header("Dataset Information: ")
    st.write("Images of 28 x 28 format belonging" 
             "to A-Z alphabet or 0-9 digits - Total Classes=36")
    st.write(
        "MODEL: VGG11"
    )
    img = st.file_uploader("Please upload Image",
                            type=["jpeg", "jpg", "png"])
    st.write("Uploaded Image")
    
    try:
        img = Image.open(img)
        st.image(img)

        st.write("Pre Processing...")
        img = process_image(img)
        img = torch.from_numpy(img).float()

        st.write("Predicting...")
        pred_res = prediction_result(model, img)
        predicted_class = pred_res["class"]

        st.write(f'Predicted class: {predicted_class}')
        st.progress(100)
    except AttributeError:
        st.write("No Image Selected")

if __name__ == '__main__':
    main()