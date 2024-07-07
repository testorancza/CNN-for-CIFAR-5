from pathlib import Path
import streamlit as st
from Model import *
import time
import cv2
import pickle

with open(str(Path(__file__).parent) + '/Model/model_params.pickle', 'rb') as file:
    params = pickle.load(file, encoding='latin1')

with open(str(Path(__file__).parent) + '/Model/image_preparation.pickle', 'rb') as file2:
    image_preparation = pickle.load(file2, encoding='latin1')

def loading_bar(description):
    progress_text = description
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

def prepare_image(image):
    image = cv2.resize(image, (32, 32)) / 255.0
    image = (image - image_preparation['mean']) / image_preparation['std']
    image = image.transpose(2, 0, 1)
    return image

def load_model():
    model = CNN()
    model.params = params
    return model