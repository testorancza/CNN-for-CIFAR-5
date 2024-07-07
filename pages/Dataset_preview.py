from pathlib import Path
import streamlit as st
from Menu import menu
import pickle
from PIL import Image

st.set_page_config(page_title='Podgląd zbioru danych', page_icon='🗃️', layout='wide')

menu()

st.title('Podgląd zbioru danych')
st.markdown('Zobacz jak wyglądają przykładowe obrazki w każdej z 5 klas.')
st.write('')

with open(str(Path(__file__).parents[1]) +'/Dataset/dataset.pickle', 'rb') as file:
    dataset = pickle.load(file, encoding='latin1')

title = ['Samolot', 'Rakieta', 'Statek', 'Czołg', 'Pociąg']

example_images = [[4, 272, 178, 232, 23], [160, 270, 69, 123, 156], [230, 231, 308, 287, 59]]

for id, column in enumerate(example_images):
    cols = st.columns(5, gap='large')
    for i in range(len(title)):
        with cols[i]:
            if not id:
                st.subheader(title[i])
            st.image(Image.fromarray(dataset['x_train'][example_images[id][i]]), width=200)
    st.write('')
