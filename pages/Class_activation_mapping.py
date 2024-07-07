from pathlib import Path
import streamlit as st
from Menu import menu
from tools import *
from PIL import Image
import cv2
import streamlit_shadcn_ui as ui

st.set_page_config(page_title='Mapy aktywacji klas', page_icon='🗺️')

menu()

st.title('Mapy aktywacji klas')
st.markdown('Mapy aktywacji klas służą do wizualizacji obszarów obrazu, które model wykorzystuje podczas dokonywania predykcji.')

labels = ['Rocket', 'Tank', 'Train', 'Airplane', 'Ship']
labels_pl = ['Rakieta', 'Czołg', 'Pociąg', 'Samolot', 'Statek']

cols = st.columns(2, gap='medium')
description = False

with cols[0]:
    option = st.selectbox('Wybierz klasę', options=labels_pl, index=None,
                          placeholder='Wybierz klasę')
    if option:
        st.header('Przykładowy obraz')
        image = Image.open(str(Path(__file__).parents[1]) + '/pages/Class_activation_mapping/Examples/{}.jpg'.format(labels[labels_pl.index(option)]))
        st.image(image.resize((750,750)), width=350)

        if ui.button(text='Generuj mapę aktywacji', key='clk_btn'):
            loading_bar('Generowanie w toku')

            image_for_model = prepare_image(np.array(image))

            model = load_model()
            scores = model.forward_pass_to_softmax(np.expand_dims(image_for_model, axis=0))
            predicted_class = np.argmax(scores, axis=1)[0]

            _, gradients = model.loss_function(x=np.expand_dims(image_for_model, axis=0), y=predicted_class)

            pooled_grads = np.mean(gradients["w1"], axis=(1, 2, 3))

            feature_maps, _ = conv2d_forward(np.expand_dims(image_for_model, axis=0), model.params['w1'],
                                             model.params["b1"], cnn_params={'stride': 1, 'pad': 1})

            for i in range(feature_maps.shape[1]):
                feature_maps[0][i][:][:] *= pooled_grads[i]

            heatmap = np.mean(feature_maps[0], axis=0)
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap)

            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)

            hif = 0.7

            resized_image = np.array(image.resize((32, 32)))

            superimposed_img = hif * heatmap + resized_image * 0.3
            superimposed_img = np.uint8(superimposed_img)

            with cols[1]:
                empty_slot = [st.write('') for _ in range(5)]
                st.header('Mapa aktywacji')

                activation_class_map = Image.fromarray(superimposed_img)
                st.image(activation_class_map, width=350)
                description = True

                cv2.imwrite(str(Path(__file__).parents[1]) + '/pages/Class_activation_mapping/CAM.jpg', cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
                with open(str(Path(__file__).parents[1]) + '/pages/Class_activation_mapping/CAM.jpg', 'rb') as image:
                    button = st.download_button('Pobierz mapę aktywacji', data=image, file_name='CAM.png', mime='image/png')

if description:
    with st.expander('Interpretacja mapy'):
        color_scale = Image.open(str(Path(__file__).parents[1]) + '/pages/Class_activation_mapping/colorscale_rainbow.jpg')
        st.image(color_scale, width=325)

        st.caption('Użyta mapa kolorów')
        st.markdown('Kolor na mapie reprezentuje istotność obszaru podczas klasyfikacji. Im jaśniejszy kolor tym bardziej istotny obszar.')



