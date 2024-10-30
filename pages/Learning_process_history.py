from pathlib import Path
import streamlit as st
from Menu import menu
import pickle
import plotly.graph_objs as go
import streamlit_shadcn_ui as ui

st.set_page_config(page_title='Wykresy procesu uczenia', page_icon='📊', layout='wide')

menu()

st.title('Wykresy procesu uczenia')
st.markdown('Zobacz jak przebiegał proces uczenia się modelu analizując wykresy dokładności oraz funkcji straty.')

with open(str(Path(__file__).parents[1]) + '/Model/model_history.pickle', 'rb') as file:
    history = pickle.load(file, encoding='latin1')

cols = st.columns(2, gap='large')

with cols[0]:
    fig = go.Figure()
    fig.add_trace(go.Line(x=list(range(1,len(history['train_accuracy_history'])+1)),
                             y=history['train_accuracy_history'],
                             name='Zbiór treningowy',
                             line=dict(color='blue'),
                          mode='lines+markers'))
    fig.add_trace(go.Line(x=list(range(1,len(history['validation_accuracy_history'])+1)),
                             y=history['validation_accuracy_history'],
                             name='Zbiór walidacyjny',
                             line=dict(color='orange'),
                            mode='lines+markers'))
    fig.update_layout(
        xaxis=dict(showline=True, showgrid=True),
        yaxis=dict(dtick=0.1),
        title='Dokładność na zbiorze treningowym i walidacyjnym w zależności od epoki',
        xaxis_title='Numer epoki',
        yaxis_title='Dokładność',
        width=700,
        height=600
    )
    st.plotly_chart(fig)
    #fig.write_image(str(Path(__file__).parents[1]) + '/pages/Learning_process_history/Accuracy_history.jpg')

    ui.metric_card(title='Największa dokładność', content=str(round(100*max(history['validation_accuracy_history']), 2)) + '%',
                   description='Dokładność informuje o tym, jaka część wszystkich predykcji modelu jest poprawna.')


with cols[1]:
    fig = go.Figure()
    fig.add_trace(go.Line(x=list(range(1,len(history['train_loss_history'])+1)),
                             y=history['train_loss_history'],
                             name='Zbiór treningowy',
                             line=dict(color='blue'),
                            mode='lines+markers'))
    fig.add_trace(go.Line(x=list(range(1,len(history['validation_loss_history'])+1)),
                             y=history['validation_loss_history'],
                             name='Zbiór walidacyjny',
                             line=dict(color='orange'),
                            mode='lines+markers'))
    fig.update_layout(
        xaxis=dict(showline=True, showgrid=True),
        yaxis=dict(dtick=0.25),
        title='Funkcja straty na zbiorze treningowym i walidacyjnym w zależności od epoki',
        xaxis_title='Numer epoki',
        yaxis_title='Wartość funkcji straty',
        width=700,
        height=600
    )
    st.plotly_chart(fig)
    
    #fig.write_image(str(Path(__file__).parents[1]) + 'pages/Learning_process_history/Loss_history.jpg')

    ui.metric_card(title='Najmniejsza strata', content=round(min(history['validation_loss_history']), 2), description='Funkcja straty służy do oceny błędu, który model popełnia na zbiorze.')


with st.expander("Wnioski", expanded=True):
        st.write("Patrząc na powyższe wykresy można zauważyć, że mamy do czynienia ze zjawiskiem przeuczenia modelu. "
                 "Sieć zbyt mocno dopasowała się do danych treningowych nie poprawiając tym samym, w kolejnych epokach, dokładności predykcji na zbiorze walidacyjnym. ")

        st.write("Dokonując ogólnej oceny modelu warto jednak zdawać sobie sprawę z ograniczeń związanych z jego implementacją oraz samym zbiorem danych. CIFAR-5 prezentując środki transportu, będące złożonymi obiektami, "
                 "wykorzystuje do tego celu obrazy o niskiej rozdzielczości. Przekłada się to na zmniejszenie szczegółowości obiektów, ograniczając tym samym ilość dostępnych dla sieci informacji.  "
                 "Modele wykorzystujące obrazy małych rozmiarów mają tendencję do zapamiętywania szczegółów danych treningowych, zamiast nauki bardziej ogólnych wzorców. "
                 "Stosowanie w tym przypadku dużej liczby warstw konwolucyjnch może doprowadzić do zredukowania mapy cech do zbyt małego rozmiaru, co uniemożliwi skuteczne uczenie się bardziej złożonych wzorców. ")

        st.write("W kontekście doboru architektury sieci kluczowe było zachowanie kompromisu między czasem trenowania, a jakością uzyskiwanych wyników. "
                 "Skorzystanie z własnej implementacji konwolucyjnej sieci neuronowej, zbudowanej bez szczególnego nacisku na optymalizację, wymagało przemyślanego balansu między złożonością modelu, a jego efektywnością obliczeniową.")
