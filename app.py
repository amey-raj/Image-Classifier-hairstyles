from pathlib import Path
from fastai.vision.all import *
from fastai.vision.widgets import *

import streamlit as st
import pathlib
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

plt =platform.system()
if plt == "Windows": pathlib.WindowsPath = pathlib.PosixPath


st.header("Multi Hair Classification")


# custom function to get labels as a list of strings
def parent_label_multi(o):
    return [Path(o).parent.name]


# Load Model #
path = Path()
learn_inf = load_learner(path/'mullti_hair_classifier_second_at.pkl')

# File uploader #
try:
    uploaded_file = st.file_uploader(
        "Upload Files", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        image_file = PILImage.create((uploaded_file))

# Display uploaded image #
    st.image(image_file.to_thumb(400, 400), caption='Uploaded Image')

# make prediction and display result #
    if st.button('Classify'):
        pred, pred_idx, probs = learn_inf.predict(image_file)
        st.write(f'Prediction:\n {pred}')
        st.write(f'Probability:\n {probs[pred_idx]}')
    else:
        st.write(f'Click the button to classify')

except NameError:
    # error will only be visible in the comand line #
    print("Name error is caught")
