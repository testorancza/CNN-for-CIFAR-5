from pathlib import Path
import streamlit as st
from Menu import menu
import pickle
from Model import *
import numpy as np
import plotly.express as px
from math import ceil, sqrt

st.set_page_config(page_title='Filtry', page_icon='🎞️', layout='wide')

menu()

with open(str(Path(__file__).parents[1]) + '/Model/model_params.pickle', 'rb') as file:
    params = pickle.load(file, encoding='latin1')

def plot_filters(x_input):
    N, H, W, C = x_input.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + 1 * (grid_size - 1)
    grid_width = W * grid_size + 1 * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C)) + 255
    indices = np.arange(N)
    next_idx = 0

    x_coords = np.tile(np.arange(grid_size) * (W + 1), grid_size)
    y_coords = np.repeat(np.arange(grid_size) * (H + 1), grid_size)

    for i in range(N):
        if next_idx < N:
            img = x_input[next_idx]
            low, high = np.min(img), np.max(img)
            grid[y_coords[i]:y_coords[i]+H, x_coords[i]:x_coords[i]+W] = 255.0 * (img - low) / (high - low)
            next_idx += 1

    return grid


cols = st.columns(7)

with cols[1]:
    st.title('Filtry')
    st.markdown('Podgląd filtrów modelu przed oraz po treningu.')

cols2 = st.columns(2)

with cols2[0]:
    model = CNN()

    filters = plot_filters(model.params['w1'].transpose(0, 2, 3, 1))
    fig = px.imshow(filters.astype('uint8'), title='Filtry przed treningiem', width=550, height=550)
    fig.update_xaxes(showline=False, visible=False)
    fig.update_yaxes(showline=False, visible=False)
    st.plotly_chart(fig)
    # fig.write_image(str(Path(__file__).parents[1]) + '/pages/Filters/Initialized_filters.jpg')

with cols2[1]:
    model = CNN()
    model.params = params

    filters = plot_filters(params['w1'].transpose(0, 2, 3, 1))
    fig = px.imshow(filters.astype('uint8'), title='Filtry po treningu', width=550, height=550)
    fig.update_xaxes(showline=False, visible=False)
    fig.update_yaxes(showline=False, visible=False)
    st.plotly_chart(fig)
    # fig.write_image(str(Path(__file__).parents[1]) + '/pages/Filters/Trained_filters.jpg')

cols3 = st.columns(3)

with cols3[1]:
    with st.expander('Znaczenie filtrów'):
        st.markdown('Filtry reprezentują lokalne wzorce obecne w danych wejściowych. '
                    'W zależności od warstwy wykrywają proste kształty takie jak krawędzie lub rogi w obrazie'
                    ' jak również bardziej złożone cechy, takie jak koła, twarze, czy inne obiekty.'
                    'Filtry zostały zainicjowane w sposób losowy i dostosowywane w trakcie procesu uczenia,'
                    ' aby najlepiej pasować do danych treningowych.')
