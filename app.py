"""Точка входа Streamlit: streamlit run app.py"""
import streamlit as st

from ui import pages

st.set_page_config(page_title="ILP-диагностика конфигурации", page_icon="🧩", layout="wide")
pages.init_state()

with st.sidebar:
    st.title("ILP Prolog")
    page = st.radio("Раздел", pages.PAGES)
    st.divider()
    st.session_state.swipl_path = st.text_input(
        "Путь к swipl (необязательно)", st.session_state.swipl_path,
        help="Оставьте пустым, если swipl есть в PATH. Пример: C:\\Program Files\\swipl\\bin\\swipl.exe")
    _, _err, _info = pages.engine_status()
    if _info:
        st.success("SWI-Prolog найден")
    else:
        st.error("SWI-Prolog не найден")

{
    "Обзор": pages.page_overview,
    "Данные": pages.page_data,
    "Обучение ILP": pages.page_train,
    "Диагностика": pages.page_diagnosis,
    "Пакетная диагностика": pages.page_batch,
    "Эксперимент": pages.page_experiment,
    "Экспорт / О проекте": pages.page_export,
}[page]()
