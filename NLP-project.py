import streamlit as st

# Заголовок страницы
st.title('Natural Language Processing Project Dashboard')

# Введение
st.header('Welcome to the NLP Project Dashboard!')
st.write(
    "This multipage application demonstrates the capabilities of various natural language processing models. "
    "Here, you'll find tools for text classification, toxicity detection, and text generation, each tailored for specific tasks."
)

# Обзор проекта
st.header('Project Overview')

st.write(
    "In this project, we explore and implement different NLP techniques. The main pages of the application include:"
)

# Список страниц
st.subheader('1. Movie Review Classification')
st.write(
    "This page allows you to classify a movie review based on user input. The dataset, which is highly imbalanced, "
    "consists of reviews from KinoPoisk. The task is to predict the review's sentiment using the following models:\n"
    "- A classic ML algorithm trained on BagOfWords/TF-IDF features.\n"
    "- An RNN or LSTM model, preferably with attention mechanisms.\n"
    "- A BERT-based model.\n"
    "For each model, you'll see the prediction results along with the time taken for inference. Additionally, "
    "a comparative table displaying the f1-macro metric for all classifiers is provided."
)

st.subheader('2. Toxicity Detection of User Messages')
st.write(
    "On this page, we use the `rubert-tiny-toxicity` model to assess the toxicity level of a user-submitted message. "
    "This tool is designed to quickly and accurately flag toxic content."
)

st.subheader('3. Text Generation with a GPT Model')
st.write(
    "This section enables users to generate text based on a custom prompt using a GPT model. You can control the length of the generated sequence, "
    "the number of generations, and fine-tune parameters like temperature or top-k/p to refine the output."
)