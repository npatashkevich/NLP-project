import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Путь к директории, где сохранена обученная модель
model_directory = './models'

# Загрузка модели и токенизатора из указанной директории
tokenizer = BertTokenizer.from_pretrained(model_directory)
model = BertForSequenceClassification.from_pretrained(model_directory)

# # Загрузка модели и токенизатора
# tokenizer = BertTokenizer.from_pretrained("cointegrated/rubert-tiny-toxicity")
# model = BertForSequenceClassification.from_pretrained("cointegrated/rubert-tiny-toxicity")

# Перевод модели в режим оценки
model.eval()

# Функция для оценки токсичности
def predict_toxicity(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return probabilities[0][1].item()  # Вероятность токсичности

# Streamlit интерфейс
st.title("Оценка степени токсичности")

# Поле ввода для пользовательского сообщения
user_input = st.text_area("Введите текст для оценки токсичности:")

# Кнопка для оценки
if st.button("Оценить токсичность"):
    if user_input:
        toxicity_score = predict_toxicity(user_input)
        st.write(f"Вероятность токсичности: {toxicity_score:.2f}")
    else:
        st.write("Пожалуйста, введите текст для оценки.")