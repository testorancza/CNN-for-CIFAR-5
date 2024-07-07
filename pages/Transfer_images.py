from pathlib import Path
import streamlit as st
import os
from Menu import menu
import pickle
from tools import *
from bing_image_downloader import downloader
import time
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title='Przekaż obrazy', page_icon='📥', layout='wide')

menu()

labels = ['Rocket', 'Tank', 'Train', 'Airplane', 'Ship']
labels_pl = ['Rakieta', 'Czołg', 'Pociąg', 'Samolot', 'Statek']

cols = st.columns(2, gap='large')

with cols[0]:
    st.title('Pozyskaj obraz')
    option = st.selectbox('Zdjęcie której klasy chcesz pobrać z Internetu?', options=labels_pl, index=None, placeholder='Wybierz klasę')
    if option:
        loading_bar('Pobieranie obrazka z Internetu')
        # downloader.download(query=labels[labels_pl.index(option)], limit=1, output_dir='Downloaded_images',
        #                     adult_filter_off=True, force_replace=False,timeout=60, verbose=False)
        path = str(Path(__file__).parents[1]) + '/pages/Transfer_images/Downloaded_images/{}/Image_1.jpg'.format(labels[labels_pl.index(option)])

        st.image(path)

        image = np.array(Image.open(path))
        image_for_model = prepare_image(image)

        loading_bar('Predykcja w toku')
        model = load_model()
        scores = model.forward_pass_to_softmax(np.expand_dims(image_for_model, axis=0))

        if labels[np.argmax(scores, axis=1)[0]] == labels[labels_pl.index(option)]:
            st.success('Model poprawnie sklasyfikował obraz.', icon="✅")
        else:
            st.error('Model sklasyfikował obraz jako {}'.format(labels_pl[np.argmax(scores, axis=1)[0]]), icon="🚨")

with cols[1]:
    st.title('Wczytaj obraz')
    upload_file = st.file_uploader('Przekaż obraz')
    if upload_file:
        bytes_data = upload_file.getvalue()
        image_array = np.frombuffer(bytes_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.image(image)

        image_for_model = prepare_image(image)

        loading_bar('Predykcja w toku')
        model = load_model()
        scores2 = model.forward_pass_to_softmax(np.expand_dims(image_for_model, axis=0))

        st.info('Model sklasyfikował obraz jako {}'.format(labels_pl[np.argmax(scores2, axis=1)[0]]), icon="ℹ️")
