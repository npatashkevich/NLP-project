import streamlit as st


st.title('Данные по обучению классификатора методом TF-IDF')
st.text('Выборка сильно несбалансирована')
st.image('images/dmche01_unbalanced.png')
st.text('Была сбалансирована по минимальной из трех категорий')
st.image('images/dmche02_balanced.png')
st.text('PipeLine обучения модели')
st.image('images/dmche03_pipeline.png')
st.text('Результирующие метрики качества модели')
st.image('images/dmche04_metrics.png')