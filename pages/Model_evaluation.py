from pathlib import Path
import streamlit as st
from Menu import menu
import pickle
from Model import *
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score
import plotly.express as px
import streamlit_shadcn_ui as ui

st.set_page_config(page_title='Ocena modelu', page_icon='📐', layout='wide')

menu()

st.title('Ocena modelu')
st.markdown('Dokonaj oceny jakości modelu na zbiorze testowym korzystając z popularnych metryk.')

cols = st.columns(2, gap='large')

with cols[0]:
    st.subheader("Ocena ogólna")
    st.write("Popularną metryką oceny jakości klasyfikatora jest dokładność, która informuje o tym jaka część wszystkich predykcji modelu jest poprawna.")

    with open(str(Path(__file__).parents[1]) + '/Dataset/dataset_for_model.pickle', 'rb') as file:
        dataset = pickle.load(file, encoding='latin1')

    with open(str(Path(__file__).parents[1]) + '/Model/evaluation_results.pickle', 'rb') as file2:
        evaluation_results = pickle.load(file2, encoding='latin1')

    labels = ['Rakieta', 'Czołg', 'Pociąg', 'Samolot', 'Statek']

    ui.metric_card("Dokładność", content=str(round(evaluation_results['acc'] * 100, 2)) + '%')

    cm = confusion_matrix(dataset['y_test'], evaluation_results['y_pred'])

    fig = px.imshow(cm.tolist(), text_auto=True, x=labels, y=labels, width=750, height=750,
                    labels=dict(x="Klasa prawdziwa", y="Klasa przewidywana", color='Liczność'),
                    title='Macierz pomyłek')
    # fig.write_image(str(Path(__file__).parents[1]) + '/pages/Model_evaluation/Confusion_matrix.jpg')
    st.plotly_chart(fig)

    with st.expander("Wnioski", expanded=False):
        st.write("Pod względem ilości poprawnych klasyfikacji, model najlepiej radzi sobie z obrazami czołgów, natomiast najgorzej z obrazami pociągów. "
                 "Najczęściej popełnianym błędem jest niepoprawna klasyfikacja czołgu jako pociąg. "
                 "Cieszy natomiast fakt niskiego odsetka pomyłek klas pod względem rodzaju trasportu (lądowy, powietrzny oraz morski). "
                 )

with cols[1]:

    st.subheader("Ocena dla poszczególnych klas")

    st.write("Analizując konkretną klasę model traktować można jako klasyfikator binarny zgodnie z podejściem jeden kontra reszta, "
             "w którym wybraną klasę traktuje się jako pozytywną, a wszystkie pozostałe jako negatywne. Wynik predykcji klasyfikatora binarnego "
             "zakwalifikować można do jednej z 4 kategorii: ")

    st.markdown("""
    - Prawdziwie pozytywny (TP) - model sklasyfikował przykład pochodzący z klasy pozytywnej jako klasa pozytywną,
    - Prawdziwie negatywny (TN) - model sklasyfikował przykład pochodzący z klasy negatywnej jako klasę negatywną,
    - Fałszywie pozytywny (FP) - model sklasyfikowal przykład pochodzący z klasy pozytywnej jako klasę negatywną (błąd pierwszego rodzaju),
    - Fałszywie negatywny (FN) - model sklasyfikowal przykład pochodzący z klasy negatywnej jako klasę pozytywną (błąd drugiego rodzaju).
    """)

    precision = precision_score(dataset['y_test'], evaluation_results['y_pred'], average=None)
    recall = recall_score(dataset['y_test'], evaluation_results['y_pred'], average=None)
    f_1 = fbeta_score(dataset['y_test'], evaluation_results['y_pred'], beta=1, average=None)

    st.write(
        "W zależności od zastosowań modelu, w ocenie klasyfikatora binarnego skorzystać można z następujących metryk:")

    cols2 = st.columns(3)

    with cols2[0]:
        st.markdown("Precyzja zdefiniowana jako")
        st.latex(r'\frac{TP}{TP + FP}')

    with cols2[1]:
        st.write("Czułość zdefiniowana jako")
        st.latex(r'\frac{TP}{TP + FN}')

    with cols2[2]:
        st.write("F1 jako średnia harmoniczna precyzji oraz czułości")
        st.latex(r'\frac{2TP}{2TP + FP + FN}')


    st.divider()

    selected_class = st.selectbox('Wybierz klasę aby poznać wartość wymienionych metryk', options=labels, index=None,
                                  placeholder='Wybierz klasę')

    cols3 = st.columns(3)

    if selected_class is not None:
        with cols3[0]:
            ui.metric_card("Precyzja", content=str(round(precision[labels.index(selected_class)] * 100, 1)) + '%')
        with cols3[1]:
            ui.metric_card("Czułość", content=str(round(recall[labels.index(selected_class)] * 100, 1)) + '%')
        with cols3[2]:
            ui.metric_card("F1", content=str(round(f_1[labels.index(selected_class)] * 100, 1)) + '%')


    with st.expander("O metryce F1"):
        st.write("Metryka F1 łącząc w sobie precyzję oraz czułość umożliwia trafną ocenę modeli w "
                 "przypadku niezbilanowanych klas jak również wykrywa sytuacje, w których model ignoruje jeden z typów błędów na rzecz poprawy drugiego. ")





