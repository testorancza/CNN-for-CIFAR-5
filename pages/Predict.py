from pathlib import Path
import streamlit as st
from Menu import menu
import pickle
from tools import *
import numpy as np
from PIL import Image
import cv2
import plotly.graph_objects as go
import streamlit_shadcn_ui as ui

st.set_page_config(page_title='Przewiduj', page_icon='💭', layout='wide')

menu()

with open(str(Path(__file__).parents[1]) + '/Dataset/dataset.pickle', 'rb') as file2:
    dataset = pickle.load(file2, encoding='latin1')

with open(str(Path(__file__).parents[1]) + '/Dataset/dataset_for_model.pickle', 'rb') as file3:
    dataset_for_model = pickle.load(file3, encoding='latin1')

labels = ['Rakieta', 'Czołg', 'Pociąg', 'Samolot', 'Statek']

cols = st.columns(2)

with cols[0]:
    st.title('Przewiduj')
    st.markdown('Dokonaj predykcji obrazka ze zbioru testowego przy użyciu wytrenowanego modelu.')
    if ui.button('Przewiduj', key='clk_btn'):

        selected_image = np.random.randint(0, len(dataset['x_test']))

        x_for_model = dataset_for_model['x_test'][selected_image]
        x = dataset['x_test'][selected_image]
        y = dataset['y_test'][selected_image]

        cols2 = st.columns(4)

        with cols2[1]:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            image_sharpened = cv2.filter2D(x, -1, kernel)
            st.image(Image.fromarray(image_sharpened), caption='Zdjęcie wyostrzone dla lepszej widoczności', width=350)
            st.write('')

        st.info('Prawdziwa klasa to {}'.format(labels[y]), icon="ℹ️")

        loading_bar('Predykcja w toku')
        model = load_model()
        scores = model.forward_pass_to_softmax(np.expand_dims(x_for_model, axis=0))

        if labels[np.argmax(scores, axis=1)[0]] == labels[y]:
            st.success('Model poprawnie sklasyfikował obraz.', icon='✅')
        else:
            st.error('Model sklasyfikował obraz jako {}'.format(labels[np.argmax(scores, axis=1)[0]]), icon='🚨')

        with cols[1]:
            st.title('Wyniki')
            st.markdown('Zobacz jak pewny w klasyfikacji był model.')

            softmax_output = np.exp(scores) / np.sum(np.exp(scores))

            fig = go.Figure(data=[go.Pie(labels=labels, values=softmax_output.flatten().tolist(), hole=0.5)])
            fig.update_layout(title='Procentowa przynależność obrazka do każdej z 5 klas')
            st.plotly_chart(fig)

            with st.expander("Jak model dokonuje predykcji?"):
                st.markdown('Mając wyniki przynależności obrazka do każdej z klas model uznaje, '
                            'że obrazek należy do klasy o największym wyniku przynależności.')





