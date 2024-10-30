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

st.title('Filtry')
st.markdown('''Istotnym elementem sieci konwolucyjnej są filtry, które umożliwiają modelowi detekcję cech obrazów umożliwiając ich klasyfikację.
        W zależności od rozmiaru, filtry skupiają się na kontretnych aspektach – mniejsze wychwytują drobne detale, 
        takie jak krawędzie, podczas gdy większe analizują bardziej złożone struktury i szersze konteksty obrazu.''')

cols = st.columns(2)

with cols[0]:
    st.subheader("Filtry przed treningiem")
    st.write('''Podczas inicjalizacji modelu, filtry dobierane są w sposób losowy przy użyciu algorytmu Glorota, 
            tak aby zminimalizować problemy związane z zanikaniem gradientu podczas wstecznej propagacji błędu w tej warstwie.
            W kolejnych iteracjach treningowych sieć na podstawie błedów, które popełnia dostosowuje filtry celem jak najlepszego dopasowania do danych.''')

    model = CNN()

    filters = plot_filters(model.params['w1'].transpose(0, 2, 3, 1))
    fig = px.imshow(filters.astype('uint8'), width=600, height=600)
    fig.update_xaxes(showline=False, visible=False)
    fig.update_yaxes(showline=False, visible=False)
    st.plotly_chart(fig)
    # fig.write_image(str(Path(__file__).parents[1]) + '/pages/Filters/Initialized_filters.jpg')


with cols[1]:
    st.subheader("Filtry po treningu")
    st.write('''Po zakończeniu uczenia filtry  prezentują  cechy obrazów, które okazały się kluczowe do poprawnej klasyfikacji przykładów. W przypadku modeli o większej liczbie warstw konwolucjnych,
    filtry w kolejnych warstwach uczą się coraz bardziej złożonych cech – zaczynając od prostych krawędzi i konturów w początkowych warstwach, po bardziej abstrakcyjne i złożone wzorce w kolejnych.''')

    model = CNN()
    model.params = params

    filters = plot_filters(params['w1'].transpose(0, 2, 3, 1))
    fig = px.imshow(filters.astype('uint8'), width=600, height=600)
    fig.update_xaxes(showline=False, visible=False)
    fig.update_yaxes(showline=False, visible=False)
    st.plotly_chart(fig)
    # fig.write_image(str(Path(__file__).parents[1]) + '/pages/Filters/Trained_filters.jpg')
