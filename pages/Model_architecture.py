from pathlib import Path
import streamlit as st
from Menu import menu
import streamlit_shadcn_ui as ui

st.set_page_config(page_title='Architektura i parametry modelu', page_icon='🏗️', layout='wide')

menu()

cols = st.columns(3)
with cols[1]:
    st.title('Architektura i parametry modelu')
    st.write('Poznaj architekturę sieci, odkrywając znaczenie poszczególnych warstw oraz zastosowanych rozwiązań.')

cols2 = st.columns([1, 2, 1])
with cols2[1]:
    st.image(str(Path(__file__).parents[1]) + '/pages/Model_architecture/Architecture.jpg', use_column_width=True)

st.markdown(
    """
    <div style="text-align: center; font-size:16px;">
        Opracowanie własne
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

cols3 = st.columns([1, 3, 1])

with cols3[1]:
    convolution, pooling, dense, activation_function, loss_function, optimizer, regularization, learning_process = st.tabs(
        ["Warstwa konwolucyjna", "Warstwa pooling'owa", "Warstwa gęsta", "Funkcja aktywacji", "Funkcja straty",
         "Optymalizator", "Regularyzacja", "Proces uczenia"])

    with convolution:
        st.write("Celem warstwy konwolucyjnej jest aplikacja filtrów na obrazie, umożliwiając ekstrakcję cech. W zależności od rozmiaru, "
                 "filtry skupiają się kontretnych aspektach – mniejsze wychwytują drobne detale, takie jak krawędzie, podczas gdy większe "
                 "analizują bardziej złożone struktury i szersze konteksty obrazu. "
                 "Każda warstwa filtra bierze udział w konwolucji z odpowiednim kanałem obrazu, a wyniki tych operacji są następnie sumowane, tworząc macierz nazywaną mapą cech. ")

        st.write("Do istotych parametrów tej warstwy należy:")
        st.markdown("""
                    - **padding** - określa liczbę pikseli dodawanych wokół krawędzi obrazu wejściowego przed wykonaniem operacji konwolucji, 
                    - **stride** - określa liczbę pikseli, o jaką porusza się jednorazowo filtr.
                    """)
        st.write("Dobór powyższych parametrów ma bezpośredni wpływ na rozmiar otrzymywanej mapy cech. ")
        ui.metric_card(title='Wybrana konfiguracja', content='16 filtrów w rozmiarze 3x3x3 pikseli', description='stride: 1, padding: 1')
    with pooling:
        st.write("Zadaniem warstwy pooling'owej jest ekstrakcja najważniejszych elementów mapy cech. "
                 "W odróżnieniu od warstwy konwolucyjnej, zamiast filtra mamy tutaj do czynienia z oknem, "
                 "które porusza się po mapie cech, zależnie od dobranych parametrów padding oraz stride."
                 " Wyróżniamy dwa najpopularniejsze rodzaje pooling'u:")
        st.markdown("""
                    - **Max Pooling** - zwraca maksymalną wartość spośród elementów mapy cech znajdujących się w oknie,
                    - **Average Pooling** - zwraca średnią arytmetyczną elementów mapy cech znajdujących się w oknie.
                    """)
        st.write("Operacja pooling'u redukuje rozmiar mapy cech w zależności wielkości okna, przyczyniając się do skrócenia czasu trenowania modelu.")
        ui.metric_card(title='Wybrana konfiguracja', content='Max Pooling', description='rozmiar okna 2x2 pikseli, stride: 2, padding: 0')
    with dense:
        st.write("W warstwie gęstej następuje połączenie informacji pochodzących z poprzednich warstw oraz przygotowanie do ostatecznej predykcji. "
                 "Stosowane w tych warstwach nieliniowe funkcje aktywacji, takie jak ReLU wprowadzają nieliniowość do modelu, umożliwiając sieci naukę bardziej złożonych zależności i wzorców w danych."
                 " Dodając warstwy gęste należy pamiętać, aby ostatnia z nich posiadała dokładnie tyle neuronów, co liczba przewidywanych klas. ")
        ui.metric_card(title='Wybrana konfiguracja', content='Warstwa gęsta zawierająca 256 neuronów',
                       description='funkcja aktywacji: ReLU')
    with activation_function:
        st.write("Funkcja aktywacji umożliwia przekształcenie informacji pochodzących z warstwy gęstej w ostateczną predykcję. "
                 "W przypadku klasyfikacji wieloklasowej popularną funkcją aktywacji jest Softmax, "
                 "który przekształca wyniki modelu na rozkład prawdopodobieństwa, umożliwiając ich łatwą intepretację. "
                 "Softmax dobrze uwypukla nawet małe różnice między wartościami, czyniąc klasy z wyższymi prawdopodobieństwami bardziej wyraźnymi. "
                 "Ponadto dobrze współpracuje z funkcji straty zwaną entropią krzyżową, która optymalizuje model, zwiększając prawdopodobieństwo dla prawdziwej klasy i zmniejszając dla innych. ")
        ui.metric_card(title='Wybrana konfiguracja', content='Softmax')
    with loss_function:
        st.write("Funkcja straty służy do oceny błędu między przewidywaniami modelu, a rzeczywistymi etykietami. "
                 "Umożliwia to, za sprawą optymalizatora, wyznaczenie kierunku, w którym wagi powinny zostać dostosowane, aby poprawić jakość predykcji."
                 "Powszechnie stosowaną funkcją straty w problemach klasyfikacji jest entropia krzyżowa, która mierzy jak bliskie rzeczywistym etykietom są przewidywania sieci. "
                 "W przypadkach predykcji poprawnej klasie bardzo niskiego prawdopodobieństwa silnie rośnie, co wymusza korektę wag sieci. "
                 "Entropia krzyżowa nie powinna być jednak stosowana w przypadkach silnie niezrównoważonych klas, bo generując wysokie wartości może zaburzyć proces uczenia.")
        ui.metric_card(title='Wybrana konfiguracja', content='Entropia krzyżowa')
    with optimizer:
        st.write("Zadaniem optymalizatora jest minimalizacja funkcji straty, w odpowiedzi na obliczone podczas wstecznej propagacji błędu gradienty. "
                 "W praktyce przekłada się na to dobór takich wartości wag oraz filtrów, które zapewnią modelowi największą dokładność predykcji. "
                 "Odpowiedni dobór oraz konfiguracja optymalizatora mają istotny wpływ na czas trenowania modelu oraz jego efektywność. "
                 "Jednym z najpopularniejszych jest ADAM (Adaptive Moment Estimation), "
                 "który łączy w sobie zalety dwóch innych popularnych optymalizatorów - momentum oraz RMSprop. "
                 "ADAM automatycznie dopasowuje współczynnik uczenia dla każdej wagi, co sprawia, że dobrze radzi sobie zarówno z rzadkimi, jak i gęstymi gradientami. "
                 "Dzięki połączeniu momentu i średniej kwadratowej gradientów stabilizuje proces optymalizacji nawet w trudnych przesztrzeniach poszukiwań. "
                 "Ponadto dobrze sprawdza się w pracy z dużymi danymi i głębokimi sieciami neuronowymi, pozwalając na szybkie zbliżanie się do minimum funkcji straty.")
        ui.metric_card(title='Wybrana konfiguracja', content='Adaptive Moment Estimation (ADAM)')
    with regularization:
        st.write("Celem stosowania technik regularyzacji jest chęć walki ze zjawiskiem przeuczenia, w którym model zbyt dokładnie dopasowuje się do danych treningowych, "
                 "tym samym znacznie pogarszając jakość swoich predykcji na zbiorze testowym."
                 "Dzięki poprawnemu stosowaniu technik regularyzacji jesteśmy w stanie zbudować lepiej generalizujący model, czyniąc go tym samym bardziej użytecznym."
                 " Do najczęściej stosowanych technik regularyzacji należy:")
        st.markdown("""
                - **Dropout** - polega na losowym ”wyłączaniu” części neuronów w trakcie treningu, co zapobiega nadmiernemu dopasowaniu do danych treningowych. Rozwiązanie to sprawia,
                  że sieć nie jest w stanie polegać na pojedynczych neuronach, stając się tym samym zmuszona do nauki bardziej ogólnych wzorców.""")
        st.markdown("""
                - **Regularyzacja L1 oraz L2** - opiera się na dodaniu kary do funkcji straty na podstawie wartości wag modelu, co pomaga w utrzymaniu ich w rozsądnych granicach i zmniejsza ryzyko
                   nadmiernego dopasowania
                    - Regularyzacja L1 - polega na dodaniu sumy wartości bezwzglednych wag do funkcji straty
                    - Regularyzacja L2 - polega na dodaniu sumy kwadatów wag do funkcji straty
                """)
        st.markdown("""
                    - **Batch Normalization** - polega na normalizacji danych wejściowych dla każdej warstwy w trakcie procesu uczenia, co stabilizuje przepływ gradientów przez sieć. 
                    Dzięki batch normalization model staje się mniej wrażliwy na początkowe wartości wag, co zwiększa stabilność uczenia.
                    """)
        st.markdown("""
                    - **Early Stopping** - monitoruje jakość predykcji modelu na zbiorze walidacyjnym i zatrzymuje trening w momencie, 
                    gdy model przestaje poprawiać jakość przewidywań, co pomaga zapobiec nadmiernemu dopasowaniu.
                    """)
        ui.metric_card(title='Wybrana konfiguracja', content='Regularyzacja L2')

    with learning_process:
        st.write("W procesie uczenia ważny jest odpowiedni sposób przekazywania modelowi przykładów oraz określenie czasu trenowania. Za odpowiednie dostosowanie treningu odpowiada: ")
        st.markdown("""
                    - wielkość paczki - informuje o ilości przykładów, która zostaje przekazana modelowi w jednej iteracji treningu
                    - liczba epok - informuje o tym, ile razy zbiór treningowy został w pełni pokazany modelowi
                    """)
        st.write("Stosowanie paczek małej wielkości jest efektywne obliczeniowo, ale z uwagi na małą liczbę przykładów może uniemożliwić sieci poprawną generalizację. "
                 "Z kolei paczki dużej wielkości pozwalają uzyskać bardziej stabilne gradienty, wymagając przy tym większych zasobów obliczeniowych.")
        st.write("W kwestii wyboru liczby epok warto poekperymentować z jej doborem lub skorzystać z metody zwanej Early Stopping, która pozwala na zatrzymanie procesu uczenia w przypadku braku poprawy jakości modelu.")
        ui.metric_card(title='Wybrana konfiguracja', content='Liczba epok: 30, wielkość paczki: 32')