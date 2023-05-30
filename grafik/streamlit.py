from flask import *
import streamlit as st
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="web_service"
)

st.set_page_config(layout="wide", page_icon="📉", page_title="Grafik Analisis")

cur = conn.cursor()
# ----------------- Gender ------------------ #
query = "SELECT * FROM gender"
df = pd.read_sql_query(query, conn)

st.write("""## Grafik Jenis Kelamin""")
chart = pd.DataFrame(df['jenis_kelamin'].value_counts())
st.line_chart(chart)

st.write("""## Grafik Rentang Umur""")
chart = pd.DataFrame(df['rentang_umur'].value_counts())
st.bar_chart(chart)

st.write("""## Grafik Label""")
chart = pd.DataFrame(df['label'].value_counts())
st.area_chart(chart)

# ---------------- Users ------------------- #

query_users = "SELECT * FROM users"
df_users = pd.read_sql_query(query_users, conn)

st.write("""## Grafik Level""")
chart = pd.DataFrame(df_users['level'].value_counts())
st.bar_chart(chart)

st.write("""## Grafik Status Validasi""")
chart = pd.DataFrame(df_users['status_validasi'].value_counts())
st.line_chart(chart)

conn.close()