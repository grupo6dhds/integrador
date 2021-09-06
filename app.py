import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
import joblib
import lightgbm as lgb
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import eli5
from eli5.lime import TextExplainer

import praw
import re
from datetime import datetime
import twint
# import nest_asyncio
# nest_asyncio.apply()
import asyncio

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import shelve
import matplotlib.pyplot as plt
import collections
import bz2
import _pickle as cPickle

from textblob import TextBlob

def main():
    # Función principal que levanta e inicializa la app
    #
    #   Con esto se configura un titulo e icono para la página web
    #   st.set_page_config(page_title="Grupo 6 - TP 4 App", page_icon="icon_g6_tp4.png", initial_sidebar_state="auto")
    st.title('Análisis de sentimiento')
    st.title('para drogas medicinales')
    #   st.header("Digital House - Data Science - TP 4 - App Grupo 6")

    opciones = ["Text scrapping", "File uploading", "Sentiment Analysis", "About"]
    # opciones = ["File uploading", "Sentiment Analysis", "About"]
    opcion_sel = st.sidebar.selectbox("Option Menu", opciones)

    if opcion_sel == "Text scrapping":
        text_scrapping()

    if opcion_sel == "File uploading":
        file_uploading()

    if opcion_sel == "Sentiment Analysis":
        sentiment_analysis()

    if opcion_sel == "About":
        about()


def text_scrapping():
    # Función para la opción de  Text Scrapping del menú de opciones
    st.subheader("Text scrapping")
    st.write("")
    with st.form("frm_text_scrap"):
        droga = st.text_input("Nombre de la droga: ")
        busqueda = ["Reddit", "Twitter"]
        busqueda_sel = st.selectbox("Opciones de búsqueda", busqueda)
        subreddit = st.text_input("Nombre del subreddit: ")
        submit_sel = st.form_submit_button(label=" Iniciar búsqueda ")
        if submit_sel:
            if droga:
                if busqueda_sel == "Twitter":
                    st.success("Ha iniciado la búsqueda de la droga '{}' en {}". format(droga, busqueda_sel))
                    procesar_tweets(droga)
                else:
                    if subreddit:
                        st.success("Ha iniciado la búsqueda de la droga '{}' en {}". format(droga, busqueda_sel))
                        procesar_reddit(droga,subreddit)
                    else:
                        st.warning("Debe seleccionar un subreddit para realizar la búsqueda")
            else:
                st.warning("Debe seleccionar una droga para realizar la búsqueda")

    st.write("")

def file_uploading():
    # Función para la opción de File uploading del menú de opciones
    st.subheader("File uploading")
    st.write("")
    with st.form("frm_file_upload"):
        st.write("El archivo debe ser formato csv y debe contener al menos las siguientes columnas: date, review, droga")
        file_upr = st.file_uploader("Seleccione un archivo (csv) para procesar: ", type='csv')
        submit_fil = st.form_submit_button(label=" Iniciar proceso ")
        if submit_fil:
            if file_upr:
                st.success("Ha iniciado el proceso del archivo seleccionado")
                procesar_archivo(file_upr)
            else:
                st.warning("Debe seleccionar un archivo antes de iniciar el proceso")
    st.write("")

def sentiment_analysis():
    # Función para la opción de Sentiment Analysis del menú de opciones
    st.subheader("Sentiment Analysis")
    st.write("")
    with st.form("frm_file_upload"):
        text_val = st.text_area("Ingrese el texto para analizar: ")
        submit_txt = st.form_submit_button(label=" Iniciar análisis de sentimiento ")
        if submit_txt:
            if text_val:
                st.success("Ha iniciado el análisis del texto ingresado")
                procesar_frase(text_val)
            else:
                st.warning("Debe ingresar un texto antes de iniciar el análisis")
    st.write("")

def about():
    # Función para la opción de About del menú de opciones
    st.subheader("About")
    st.write("Participantes:")
    st.write("Mariana Peinado")
    st.write("Juan Boirazian")
    st.write("Jorge Corro")
    st.write("Franco Visintini")
    st.write("Federico Vessuri")

def procesar_tweets(droga):
    st.write("buscar tweets...")
    df = buscar_tweets(droga)
    if df is not None:
        st.write("dataframe de tweets:", df.head(5))
        st.write("shape: ", df.shape)
        procesar_resultados(df, droga)
    else:
        st.warning("No hay datos en el archivo para procesar")

def procesar_reddit(droga, subreddit):
    st.write("buscar en reddit...")
    df = buscar_reddit(subreddit, droga)
    if df is not None:
        st.write("dataframe de reddit:", df.head(5))
        st.write("shape: ", df.shape)
        procesar_resultados(df, droga)
    else:
        st.warning("No hay datos en el archivo para procesar")

def procesar_archivo(arch):
    # Función para cuando se ejecuta la opción de File uploading
    if arch is not None:
        df = pd.read_csv(arch)
        if df is not None:
            droga = df["droga"][0]
            st.write("dataframe del archivo:", df.head(5))
            st.write("shape: ", df.shape)
            procesar_resultados(df, droga)
        else:
            st.warning("No hay datos en el archivo para procesar")
    else:
        return None

def procesar_frase(texto):
    # Función para cuando se ejecuta la opción de Sentiment Analysis
    rev = [texto]
    # modelos = load_model("modelo_svd_cvectorizer")
    # lgbm = modelos["lgbm"]
    # svd = modelos["svd"]
    # cvect = modelos["cvectorizer"]
    # lgbm = decompress_pickle("modelo_lgbm_08")
    # svd = decompress_pickle("modelo_svd_08")
    # cvect = decompress_pickle("modelo_cvect_08")
    lgbm, svd, cvect = obtener_modelos("modelo_lgbm_08", "modelo_svd_08", "modelo_cvect_08")
    # pred = lgbm.predict(svd.transform(cvect.transform(rev)))
    pred = predecir_reviews(rev, lgbm, svd, cvect)
    # st.write("pred = ", pred)
    tb_res = TextBlob(texto).sentiment
    # st.write("tb_res: ", tb_res)
    pol = tb_res[0]
    subj = tb_res[1]
    if pred[0] == 1:
        st.success("Resultado: Satisfatorio")
    else:
        st.error("Resultado: Insatisfatorio")
    if pol > 0:
        st.write("Sentimiento: Positivo")
    elif pol == 0:
        st.write("Sentimiento: Neutro")
    else:
        st.write("Sentimiento: Negativo")
    if subj > 0:
        st.write("Subjetividad: Alta")
    elif subj == 0:
        st.write("Subjetividad: Neutra")
    else:
        st.write("Subjetividad: Baja")

def buscar_tweets(droga):
    # Función para hacer scrapping de Twitter con la librería twint
    #### BUSCO TWEETS QUE CONTENGAN LA FRASE "Droga is" y despues lo guardo en el DF Tweets_df
    c = twint.Config()
    Busqueda =   """\"""" + droga + " is" + """\""""
    c.Search = Busqueda
    c.Limit = 200
    c.Pandas = True
    c.Lang="en"
    st.write("Busqueda: ", Busqueda)
    asyncio.set_event_loop(asyncio.new_event_loop())
    twint.run.Search(c)
    Tweets_df = twint.storage.panda.Tweets_df
    Tweets_df['droga'] = droga
    Tweets_df=Tweets_df[['date','tweet','droga']]
    Tweets_df.rename(columns={'tweet': 'review'}, inplace=True)
    c.Hide_output = True
    return Tweets_df

def buscar_reddit(subredd , droga):
    # Función para hacer scrapping de Reddit 
    i=0
    column_names = ["droga", "review", "date"]
    df = pd.DataFrame(columns = column_names)
    Subreddit = subredd ### el subreddit donde quiero hacer la busqueda
    Busqueda =   """\"""" + droga + " is" + """\""""
    reddit = get_reddit_credentials()
    subR = reddit.subreddit(Subreddit)
    st.write("buscando en reddit la droga {} en el subreddit {}...".format(droga,subredd))
    resp = subR.search(Busqueda,limit=100)
    st.write("busqueda en reddit finalizada...")
    for submission in resp:
        df.at[i, 'droga'] = droga
        df.at[i, 'review'] =str(str(submission.title.encode('ascii', 'ignore').decode("utf-8")) +" "+ str(submission.selftext[:120].encode('ascii', 'ignore').decode("utf-8")))        
        df.at[i, 'date'] = datetime.utcfromtimestamp(int((submission.created_utc))).strftime('%Y-%m')
        i+=1
    return df

def get_reddit_credentials():
    return praw.Reddit(client_id='5U6IG9mVmOBz08m7gb_z8Q',client_secret='Y8yZhKAmDk6ryyEiXutrM0SVgnAMEg',username='jboirazian',password='+xj<_6$9hsZ7E)L',user_agent='jboirazian_grupo4')

def procesar_resultados(df, droga):
    clean_review = procesar_dataframe(df)
    lgbm, svd, cvect = obtener_modelos("modelo_lgbm_08", "modelo_svd_08", "modelo_cvect_08")
    clean_review_pred = predecir_reviews(clean_review, lgbm, svd, cvect)
    #st.write("Predicción de las reviews: ", clean_review_pred)
    count = collections.Counter(clean_review_pred)
    data_chart = pd.DataFrame({
    'Sentiment': ['Insatisfactorio', 'Satisfactorio'],
    'Results': [count[0], count[1]],
    })
    st.write(data_chart)
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Insatisfatorios', 'Satisfactorios'
    sizes = [count[0], count[1]]
    explode = (0, 0.1)  # sólos e hace "explode" de los "Satisfactorios"

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    #ax1.title(droga)

    st.pyplot(fig1)

def limpiar_texto(texto):
    # Función de limpieza de texto
    # poner todo en minúscula
    minuscula = texto.lower()   
    # quitar patrón de repetición
    pattern=minuscula.replace('&#039;', "'")
    # Remover caracteres especiales
    special_remove = pattern.replace(r'[^\w\d\s]',' ')    
    # Remover los caracteres no ASCII
    ascii_remove = special_remove.replace(r'[^\x00-\x7F]+',' ')    
    # Remover espacios en blanco iniciales y finales
    whitespace_remove = ascii_remove.replace(r'^\s+|\s+?$','')    
    # Replazar multiples espacios en blanco con un ùnico espacio
    multiw_remove = whitespace_remove.replace(r'\s+',' ')    
    # Replazar 2 o más puntos por 1 solo punto
    dots_remove = multiw_remove.replace(r'\.{2,}', ' ')    
    return dots_remove

def contar_palabras(s):
    return (len(s.split()))

# Esta función aplica un tokenizador, genera stems y borra stop_words
def clean_datos(review_text, tokenizer, stemmer, stopwords):
    # Función para limpiar datos y prepararlos para la predicción
    stopwords_stem = [stemmer.stem(x) for x in stopwords]    #tokens (eliminamos todos los signos de puntuación)
    words = tokenizer.tokenize(review_text)
    # stemming: raiz y minúsculas:
    stem_words = [stemmer.stem(x) for x in words]
    # eliminamos stopwords (ya pasaron por stem)
    clean_words = [x for x in stem_words if x not in stopwords_stem]
    result = " ".join(clean_words)
    return(result)

def procesar_dataframe(df):
    # Función para cuando limpiar las reviews de un dataframe
    df.review.apply(limpiar_texto)
    # Definimos tokenizador, stemmer y stop_words que utilizaremos en la función "clean_datos"
    tokenizer = RegexpTokenizer(r"\w+")
    englishStemmer = SnowballStemmer("english")
    stopwords_en = stopwords.words('english')
    # Se quitan las stopwords y se stemizan las palabras limpias de las reviews del dataframe
    clean_review = [clean_datos(x, tokenizer, englishStemmer, stopwords_en) for x in df.review]
    return clean_review

@st.cache(allow_output_mutation=True)
def obtener_modelos(file_lgbm, file_svd, file_cvect):
    lgbm = decompress_pickle(file_lgbm)
    svd = decompress_pickle(file_svd)
    cvect = decompress_pickle(file_cvect)
    return lgbm, svd, cvect

def predecir_reviews(reviews, lgbm, svd, cvect):
    # Función para obtener la predicción de las reviews
    # Se obtienen los modelos entrenados
    # modelos = load_model("modelo_svd_cvectorizer")
    # lgbm = modelos["lgbm"]
    # svd = modelos["svd"]
    # cvect = modelos["cvectorizer"]
    # lgbm = decompress_pickle("modelo_lgbm_08")
    # svd = decompress_pickle("modelo_svd_08")
    # cvect = decompress_pickle("modelo_cvect_08")
    # Se realiza la predicción de las reviews
    pred = lgbm.predict(svd.transform(cvect.transform(reviews)))
    return pred

# def load_model(file):
#     # Función para recargar un modelo entrenado [se usa la librería shelve]
#      m = shelve.open(file)
#      return m

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file + ".pbz2", "rb")
    data = cPickle.load(data)
    return data

# Se inicia la app con la función main
if __name__ == '__main__':
    main()
