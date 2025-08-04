#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import dill
import pandas as pd

# Загрузка модели
@st.cache_resource
def load_model():
    with open(r"C:\Users\BSPB\Downloads\titanic_model.sav", 'rb') as f:
        return dill.load(f)

pipe = load_model()

# Функция для предсказания
def predict_probability(input_data):
    try:
        # Преобразуем в DataFrame (важен порядок признаков!)
        df = pd.DataFrame([input_data])
        # Получаем вероятность класса 1 (например, "выжил")
        proba = pipe.predict_proba(df)[0][1]  
        return proba
    except Exception as e:
        st.error(f"Ошибка предсказания: {e}")
        return None

# Веб-интерфейс
st.title("Прогнозирование вероятности (Пример: Titanic)")
st.markdown("Введите параметры и нажмите **'Спрогнозировать'**")

# Поля для ввода (адаптируйте под ваши признаки!)
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Класс билета", [1, 2, 3], help="1 — Первый класс")
    sex = st.selectbox("Пол", ["male", "female"])
    age = st.number_input("Возраст", min_value=0, max_value=100, value=30)

with col2:
    sibsp = st.number_input("Число родственников на борту", min_value=0, max_value=10, value=0)
    parch = st.number_input("Число родителей/детей", min_value=0, max_value=10, value=0)
    fare = st.number_input("Стоимость билета", min_value=0.0, value=50.0)

# Кнопка предсказания
if st.button("Спрогнозировать вероятность"):
    input_data = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare
    }
    
    probability = predict_probability(input_data)
    if probability is not None:
        st.success(f"Вероятность: **{probability:.2%}**")
        # Визуализация
        st.progress(float(probability))
        st.markdown(f"🔴 **Интерпретация:** {'Высокий шанс' if probability > 0.5 else 'Низкий шанс'}")


# In[ ]:




