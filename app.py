from os import path, listdir
import streamlit as st
from streamlit_embedcode import github_gist
import streamlit.components.v1 as com
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pdfplumber
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords 
from string import punctuation
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.collocations import *
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu
from os import path, listdir
import glob
import pickle
from pathlib import Path
from plotly import graph_objs as go
from collections import Counter
from sklearn.metrics.pairwise import linear_kernel
from st_material_table import st_material_table
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import streamlit.components.v1 as components


# Ignore warning
st.set_option('deprecation.showPyplotGlobalUse', False)
# Set wide layout
st.set_page_config(
    page_title = 'SVIP',
    page_icon = '✅',
    layout = 'wide'
)
st.markdown("""------------------------------------------------------------------------------------------------""")
 #Gambar logo
from PIL import Image
img1 = Image.open("data/logo gandeng tiga+paripurna.png")
st.image(img1, width=900)

# SISTEM 
def main():
    #harozontal menu
    selected = option_menu(
        menu_title="SVIP (SISTEM VISUALISASI INFORMASI PASIEN) RST BHAKTI WIRA TAMTAMA",
        options=["Visualisasi Data Pasien", "Check Health Predictions","ChatSVIP"],
        icons=["map","heart", "chat-square-dots"],
        menu_icon="globe",
        default_index=0,
        orientation="horizontal",
        styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "yellow", "font-size": "25px"},
                    "nav-link": {
                        "font-size": "13px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#eee",
                    },
                    "nav-link-selected": {"background-color": "green"},
                },
        )
      
    #Exploration
    if selected == "Visualisasi Data Pasien":
        # sidebar for navigation
        with st.sidebar:
            selected = option_menu('Visualisasi Data Pasien',
                                   ['Visualisasi Peta Persebaran Pasien',
                                    'Visualisasi PPK Rujukan Pasien'
                                    ],
                                   icons=['map','bar-chart'],
                                   default_index=0)
       
     
        if (selected == 'Visualisasi Peta Persebaran Pasien'):
            st.date_input("")
            # read csv 
            df_ket = pd.read_csv("data/datapasienrst.csv")
            st.markdown("""------------------------------------------------------------------------------------------------""")
            # dashboard title
            st.title("Peta Persebaran Pasien Wilayah Semarang")
            #Bar option pilihan
            st.markdown("""------------------------------------------------------------------------------------------------""")
            # --- Information of total Data ---
            
            with open('style.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            
            df = pd.read_csv("data/datapasienrst.csv")
            
            total_pria, total_perempuan, total = st.columns(3)
            total_pria.metric("Total Jenis Kelamin Pria", df.jml_pria.sum())
            total_perempuan.metric("Total Jenis Kelamin Perempuan",  df.jml_perempuan.sum())
            total.metric("Total Keseluruan", df.jml.sum())
        
            st.markdown("""------------------------------------------------------------------------------------------------""")
            #Dhasboard peta
            com.html("""
            <html>
            <body>
            <div class='tableauPlaceholder' 
            id='viz1670645955573' 
            style='position: relative'>
            <noscript>
            <a href='#'>
            <img alt='Dashboard 4 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;B_&#47;B_16706430421050&#47;Dashboard4&#47;1_rss.png' style='border: none' />
            </a>
            </noscript>
            <object class='tableauViz'  style='display:none;'>
            <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
            <param name='embed_code_version' value='3' /> 
            <param name='site_root' value='' />
            <param name='name' value='B_16706430421050&#47;Dashboard4' />
            <param name='tabs' value='no' />
            <param name='toolbar' value='yes' />
            <param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;B_&#47;B_16706430421050&#47;Dashboard4&#47;1.png' />
            <param name='animate_transition' value='yes' />
            <param name='display_static_image' value='yes' />
            <param name='display_spinner' value='yes' />
            <param name='display_overlay' value='yes' />
            <param name='display_count' value='yes' />
            <param name='language' value='en-US' />
            </object>
            </div>                
            <script type='text/javascript'>                    
            var divElement = document.getElementById('viz1670645955573');                    
            var vizElement = divElement.getElementsByTagName('object')[0];                    
            if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='727px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                
            </script>
            </body>
            </html>
            """,height=800)
        
            st.markdown("""------------------------------------------------------------------------------------------------""")
            st.write("**Dataset**")  # add a title
            df =  pd.read_csv("data/datapasienrst1.csv")  # read a CSV file inside the 'data"
            #st.write(df)  # visualize my dataframe in the Streamlit app
            _ = st_material_table(df)
        
            st.markdown("""------------------------------------------------------------------------------------------------""")
            
        #Visualisai PPK Rujukan
        if (selected == 'Visualisasi PPK Rujukan Pasien'):
            com.html("""
            <html>
            <body>
            <div class='tableauPlaceholder' id='viz1671361769500' style='position: relative'><noscript><a href='#'><img alt='Dashboard 3 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;pp&#47;ppkrujukan&#47;Dashboard3&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='ppkrujukan&#47;Dashboard3' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;pp&#47;ppkrujukan&#47;Dashboard3&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1671361769500');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='1227px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
            </html>
            """,height=988)
        
     # Prediksi
    elif selected == "Check Health Predictions":  
        diabetes_model = pickle.load(open('models/diabetes_model.sav', 'rb'))
        heart_disease_model = pickle.load(open('models/heart_disease_model.sav','rb'))
  
        # sidebar for navigation
        with st.sidebar:
            selected = option_menu('Check Health Predictions System',
                                   ['Prediction Penyakit Diabetes',
                                    'Prediction Penyakit Jantung',
                                    'Prediction BMI'],
                                   icons=['activity','heart','person'],
                                   default_index=0)
            
        # Diabetes Prediction Page
        if (selected == 'Prediction Penyakit Diabetes'):
            with open('style.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            # page title
            st.title('Prediction Penyakit Diabetes')
            st.write('Sistem ini anda bisa gunakan untuk mengetahui hasil prediksi penyakit diabetes')
            # getting the input data from the user
            col1, col2, col3 = st.columns(3)
            with col1:
                Pregnancies = st.number_input('Jumlah Kehamilan')
            with col2:
                Glucose = st.number_input('Tingkat Glukosa')
            with col3:
                BloodPressure = st.number_input('Nilai Tekanan Darah')
            with col1:
                SkinThickness = st.number_input('Skin Thickness value/Nilai Ketebalan Kulit')
            with col2:
                Insulin = st.number_input('Tingkat Insulin')
            with col3:
                BMI = st.number_input('Nilai BMI')
            with col1:
                DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
            with col2:
                Age = st.number_input('Umur Anda')
            # code for Prediction
            diab_diagnosis = ''
            # creating a button for Prediction
            if st.button('Check Test Diabetes'):
                diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                                                           DiabetesPedigreeFunction, Age]])
                if (diab_prediction[0] == 1):
                    diab_diagnosis = 'Hasil Prediksi Anda  menderita diabetes'
                else:
                    diab_diagnosis = 'Hasil Prediksi Anda tidak menderita diabetes'
            st.success(diab_diagnosis)
            st.markdown("""------------------------------------------------------------------------------------------------""")
            
        #Prediction Penyakit Jantung
        if (selected == 'Prediction Penyakit Jantung'):
            with open('style.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            # page title
            st.title('Prediction Penyakit Jantung')
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input('Umur')
            with col2: 
                sex = st.number_input('Jenis Kelamin (1 = laki-laki; 0 = perempuan)')
            with col3:
                cp = st.number_input('Chest Pain types (0 = typical, 1 = asymptotic, 2 = nonanginal, 3 = nontypical)')
            with col1:
                trestbps = st.number_input('Resting Blood Pressure')
            with col2:
                chol = st.number_input('Serum Cholestoral in mg/dl')
            with col3:
                fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl (1 = benar; 0 = salah)')
            with col1:
                restecg = st.number_input('Resting Electrocardiographic results')
        
            with col2:
                thalach = st.number_input('Maximum Heart Rate achieved')
            with col3:
                exang = st.number_input('Exercise Induced Angina (1 = ya; 0 = tidak)')  
            with col1:
                oldpeak = st.number_input('ST depression induced by exercise')
            with col2:
                slope = st.number_input('Slope of the peak exercise ST segment')
            with col3:
                ca = st.number_input('Major vessels colored by flourosopy (0 - 3)')
            with col1:
                thal = st.number_input('thal: 1 = normal; 2 = fixed defect; 3 = reversable defect')
  
            # code for Prediction
            heart_diagnosis = ''
            # creating button Prediction

            if st.button('Check Test Prediksi Jantung'):
                heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, 
                                                                 restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
                if (heart_prediction[0] == 1):
                    heart_diagnosis = 'Anda Terdeteksi penyakit Jantung'
                else:
                    heart_diagnosis = 'Anda Tidak Terdeteksi penyakit Jantung'
            st.success(heart_diagnosis)
        
        #' Prediction BMI'
        if (selected == 'Prediction BMI'):
            # page title
            st.title('Prediction Body Mass Index atau BMI')
            st.write('Body Mass Index (BMI) atau Indeks Massa Tubuh (IMT) adalah angka yang menjadi penilaian standar untuk menentukan apakah berat badan Anda tergolong normal, kurang, berlebih, atau obesitas')
            st.write('Untuk menghitung BMI Anda, silakan isi kolom berikut:')


            weight = st.number_input("Masukkan Berat Anda dalam KG")
            height = st.number_input("Masukkan Tinggi Badan Anda dalam Meter")
            if st.button('Check Test BMI'):
                bmi = (weight/(height*height)*2)
                st.success(f"Hasil Test BMI Anda {bmi}")
            img_bmi = Image.open("data/bmi.png")
            st.image(img_bmi, width=800)
        
    # Chat
    elif selected == "ChatSVIP": 
        com.html("""
        <html>
        <body>
        <script src="https://www.gstatic.com/dialogflow-console/fast/messenger/bootstrap.js?v=1"></script>
        <df-messenger
        intent="WELCOME"
        chat-title="ChatSVIP"
        agent-id="c33cdaaf-28e2-4f15-ab78-769fe5cd0df8"
        language-code="id">
        </df-messenger>
         </body>
         </html>
          """ ,height=660
        )
        

           

if __name__ == '__main__':
    main()

 
                
                
