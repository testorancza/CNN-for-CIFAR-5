from pathlib import Path
import streamlit as st
from Menu import menu
import pickle
import plotly.graph_objs as go
import streamlit_shadcn_ui as ui

st.set_page_config(page_title='Wykresy procesu uczenia', page_icon='📊', layout='wide')

menu()

st.title('Wykresy procesu uczenia')
st.markdown('Zobacz jak przebiegał proces uczenia się modelu.')

with open(str(Path(__file__).parents[1]) + '/Model/model_history.pickle', 'rb') as file:
    history = pickle.load(file, encoding='latin1')

cols = st.columns(2, gap='large')

with cols[0]:
    fig = go.Figure()
    fig.add_trace(go.Line(x=list(range(len(history['train_accuracy_history']))),
                             y=history['train_accuracy_history'],
                             name='Zbiór treningowy',
                             line=dict(color='blue'),
                          mode='lines+markers'))
    fig.add_trace(go.Line(x=list(range(len(history['validation_accuracy_history']))),
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
        width=600,
        height=600
    )
    st.plotly_chart(fig)
    #fig.write_image(str(Path(__file__).parents[1]) + '/pages/Learning_process_history/Accuracy_history.jpg')

    ui.metric_card(title='Największa dokładność', content=str(round(100*max(history['validation_accuracy_history']), 2)) + '%',
                   description='Im większa dokładność tym lepszy model.')

with cols[1]:
    fig = go.Figure()
    fig.add_trace(go.Line(x=list(range(len(history['train_loss_history']))),
                             y=history['train_loss_history'],
                             name='Zbiór treningowy',
                             line=dict(color='blue'),
                            mode='lines+markers'))
    fig.add_trace(go.Line(x=list(range(len(history['validation_loss_history']))),
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
        width=600,
        height=600
    )
    st.plotly_chart(fig)
    #fig.write_image(str(Path(__file__).parents[1]) + 'pages/Learning_process_history/Loss_history.jpg')

    ui.metric_card(title='Najmniejsza strata', content=round(min(history['validation_loss_history']), 2), description='Im mniejsza strata tym lepszy model.')
