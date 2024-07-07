from pathlib import Path
import streamlit as st
from Menu import menu
import pickle
from Model import *
from sklearn.metrics import confusion_matrix
import plotly.express as px
import streamlit_shadcn_ui as ui

st.set_page_config(page_title='Ocena modelu', page_icon='📐')

menu()

st.title('Ocena modelu')
st.markdown('Zobacz wyniki modelu na zbiorze testowym.')

with open(str(Path(__file__).parents[1]) + '/Dataset/dataset_for_model.pickle', 'rb') as file:
    dataset = pickle.load(file, encoding='latin1')

with open(str(Path(__file__).parents[1]) + '/Model/evaluation_results.pickle', 'rb') as file2:
    evaluation_results = pickle.load(file2, encoding='latin1')

labels = ['Rakieta', 'Czołg', 'Pociąg', 'Samolot', 'Statek']

cols = st.columns(2)

with cols[0]:
    ui.metric_card("Dokładność", content=str(round(evaluation_results['acc'] * 100, 2)) + '%')
with cols[1]:
    ui.metric_card("Funkcja straty", content=round(evaluation_results['loss'],2))

cm = confusion_matrix(dataset['y_test'], evaluation_results['y_pred'])
fig = px.imshow(cm.tolist(), text_auto=True, x=labels, y=labels, width=600, height=600,
                labels=dict(x="Klasa prawdziwa", y="Klasa przewidywana", color='Liczność'),
                title='Macierz pomyłek')
#fig.write_image(str(Path(__file__).parents[1]) + '/pages/Model_evaluation/Confusion_matrix.jpg')
st.plotly_chart(fig)

with st.expander('Wnioski'):
    st.markdown('Model w największym stopniu myli się podczas klasyfikacji statku i samolotu oraz czołgu i pociągu.')

