import streamlit as st
import os
import re
import requests
import joblib
import pymorphy3
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from collections import defaultdict
import string
import zipfile
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
# model_file_path = os.path.join(current_dir, '../models/tfidf_text_classification_pipeline.pkl')
model_file_path = './models/tfidf_text_classification_pipeline.pkl'

loaded_pipeline = joblib.load(model_file_path)

st.write(model_file_path)

st.title("Классификация отзывов о фильмах по шкале Good-Neutral-Bad")
new_text = st.text_input("Введите свой отзыв для классификации:")

with st.sidebar:
    st.subheader('Примеры текста для ввода:')
    st.text('[Good]')
    st.write('Фильм впечатлил своей глубиной и неожиданными сюжетными поворотами. Актерская игра на высоте — каждый персонаж показан с такой искренностью, что легко забыть, что это всего лишь кино. Визуальные эффекты потрясают, а саундтрек идеально дополняет атмосферу. Режиссер умело сочетает драму, экшен и моменты комедии, создавая цельное произведение искусства. Этот фильм оставил неизгладимое впечатление, заставил задуматься и даже пересмотреть некоторые жизненные приоритеты. Определенно рекомендую к просмотру, особенно тем, кто любит фильмы с глубоким смыслом и харизматичными героями.')
    st.text('[Neutral]')
    st.write('Фильм оказался неплохим, но не произвел сильного впечатления. Сюжет в целом интересный, хотя местами предсказуемый. Актеры сыграли достойно, но не скажу, что кто-то особо выделился. Визуальные эффекты и музыка были на уровне, но не вызвали вау-эффекта. Временами динамика фильма казалась затянутой, и это немного снижало общее впечатление. В целом, картина подойдет для вечернего просмотра, но не думаю, что она останется в памяти надолго. Неплохой фильм, но ничего выдающегося в нем я не нашел.')
    st.text('[Bad]')
    st.write('Фильм разочаровал по всем фронтам. Сюжет оказался банальным и скучным, а развитие событий — предсказуемым. Актерская игра оставляет желать лучшего — персонажи получились плоскими и невыразительными. Визуальные эффекты и саундтрек тоже не спасли ситуацию — все выглядело дешево и неумело. Фильм явно тянули по времени, добавляя ненужные сцены и затягивая простые моменты. В итоге я просто потерял время, ожидая, что что-то изменится, но, увы, этого не произошло. Определенно не рекомендую к просмотру.')

if new_text:
    # Определяем класс и измеряем время предсказания
    start_time = time()
    predicted_class = loaded_pipeline.predict([new_text])
    end_time = time()

    # Вывод результатов классификации и времени предсказания
    st.text(f"Новый текст относится к классу: {predicted_class[0]}")
    st.text(f"Время предсказания: {end_time - start_time:.4f} секунд")

else:
    st.stop()

