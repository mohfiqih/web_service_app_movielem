# from flask import *
# import streamlit as st
# import numpy as np 
# import pandas as pd 
# import matplotlib.pyplot as plt
# import seaborn as sns

# df_gender = pd.read_csv('data/gender.csv')
# df_user = pd.read_csv('data/users.csv')

# st.set_page_config(layout="wide", page_icon="📉", page_title="Diagram Analisis")

# st.write("""## Tabel Data Gender""")
# st.write(df_gender)

# st.write("""## Tabel Sample""")
# data = df_gender.iloc[:, [1, 2, 3]]
# st.write(data)

# st.write("""## Grafik Gender""")
# gender = data.iloc[:, [0]]
# total_gender = gender.value_counts()
# st.write(total_gender)

# st.write("""## Grafik Rentang Umur""")
# rentang = data.iloc[:, [1]]
# total_rentang = rentang.value_counts()
# st.write(total_rentang)
 
# st.write("""## Grafik Label""")
# label = data.iloc[:, [2]]
# total_label = label.value_counts()
# st.write(total_label)

# st.write("""## Tabel Data Users""")
# st.write(df_user)

# st.write("""## Tabel Sample""")
# data = df_user.iloc[:, [5, 6]]
# st.write(data)

# st.write("""## Grafik Level User""")
# level_user = data.iloc[:, [1]]
# total_level_user = level_user.value_counts()
# st.write(total_level_user)

# st.write("""## Grafik Status""")
# status_validasi = data.iloc[:, [0]]
# total_status_validasi = status_validasi.value_counts()
# st.write(total_status_validasi)

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

st.set_page_config(layout="wide", page_icon="📉", page_title="Total")

cur = conn.cursor()
query = "SELECT * FROM gender"
query_users = "SELECT * FROM users"

st.write("""## Tabel Gender""")
df = pd.read_sql_query(query, conn)
st.write(df)

st.write("""## Tabel Jenis Kelamin""")
gender = df.iloc[:, [1]]
total_gender = gender.value_counts()
st.write(total_gender)

st.write("""## Tabel Rentang Umur""")
rentang = df.iloc[:, [2]]
total_rentang = rentang.value_counts()
st.write(total_rentang)

st.write("""## Tabel Label""")
label = df.iloc[:, [3]]
total_label = label.value_counts()
st.write(total_label)

# ----------- Users --------- #

st.write("""## Tabel Users""")
df_users = pd.read_sql_query(query_users, conn)
st.write(df_users)

st.write("""## Grafik Status Validasi""")
status_validasi = df_users.iloc[:, [5]]
total_status_validasi = status_validasi.value_counts()
st.write(total_status_validasi)

st.write("""## Grafik Level User""")
level_user = df_users.iloc[:, [6]]
total_level_user = level_user.value_counts()
st.write(total_level_user)

conn.close()