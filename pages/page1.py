import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import re
import requests
import joblib
import string
import zipfile
import pymorphy3
from collections import defaultdict
from time import time
from umap import UMAP
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class MyCustomTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = self.get_stopwords_list()
        self.morph = pymorphy3.MorphAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, texts, y=None):
        return [self.preprocess(text) for text in texts]

    def get_stopwords_list(self):
        # URL файла со стоп-словами
        url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt"
        # Загрузка содержимого файла
        response = requests.get(url)
        # Чтение содержимого и разбивка на строки
        stop_words = set(response.text.splitlines())
        return stop_words

    def clean(self, text):
        # Переводим текст в нижний регистр
        text = text.lower()
        # Убираем смайлики
        text = re.sub(r':[a-zA-Z]+:', '', text)
        # Убираем упоминания пользователей
        text = re.sub(r'@[\w_-]+', '', text)
        # Убираем хэштеги
        text = re.sub(r'#(\w+)', '', text)
        # Убираем цифры
        text = re.sub(r'\d+', '', text)
        # Убираем ссылки
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text)
        # Удаление английских слов
        text = ' '.join(re.findall(r'\b[а-яА-ЯёЁ]+\b', text))
        return text.strip()

    def remove_stopwords(self, text):
        # Разбиваем текст на слова
        words = text.split()
        # Фильтруем слова, оставляя только те, которые не являются стоп-словами
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        # Собираем текст из отфильтрованных слов
        filtered_text = ' '.join(filtered_words)
        return filtered_text

    def lemmatize(self, text):
        # Разбиваем текст на слова
        words = text.split()
        # Лемматизируем каждое слово
        lemmatized_words = [self.morph.parse(word)[0].normal_form for word in words]
        # Собираем текст из лемматизированных слов
        lemmatized_text = ' '.join(lemmatized_words)
        return lemmatized_text

    def preprocess(self, text):
        text = self.clean(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize(text)
        return text



# Загружаем предобученную модель
current_dir = os.path.dirname(os.path.abspath(__file__))
# model_file_path = os.path.join(current_dir, '../models/text_classification_pipeline.pkl')
model_file_path = os.path.join(current_dir, '../models/tfidf_text_classification_pipeline.pkl')
loaded_pipeline = joblib.load(model_file_path)


st.title("Классификация отзывов о фильмах по шкале Good-Neutral-Bad")
new_text = st.text_input("Введите свой отзыв для классификации")
st.text("Примеры текста для ввода:\n"
        "[Good] Никогда на душе у меня не было так спокойно, как в этот раз. Этот потрясающий фильм просто вознес мою душу "
        "на недостижимые высоты и там пребывала она в блаженстве, аки у Христа за пазухой.\n"
        "[Bad] Что за дебильный фильм? Кто вообще такое смотрит и кто вообще такое снимает? Это же просто позор!!!\n"
        "[Neutral] Ни че так фильмец) Смотреть можно. Бывает и лучше, но мы на чили чисто посмотрели")

if new_text:
    # Определяем класс и измеряем время предсказания
    start_time = time()
    predicted_class = loaded_pipeline.predict(new_text)
    end_time = time()

    # Вывод результатов классификации и времени предсказания
    st.text(f"Новый текст относится к классу: {predicted_class[0]}")
    st.text(f"Время предсказания: {end_time - start_time:.4f} секунд")

else:
    st.stop()

