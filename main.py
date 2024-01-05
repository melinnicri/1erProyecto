from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import pandas as pd
import importlib
import numpy as np
import parquet


# Usar dataframes desde parquet (consultas)
genres_year_playtime = pd.read_parquet('genres_year_playtime.parquet')
max_playtime_by_genre_year= pd.read_parquet('max_playtime_by_genre_year.parquet')
user_year_playtime = pd.read_parquet('user_year_playtime.parquet')
year_recommend_game = pd.read_parquet('year_recommend_game.parquet')
top3_recomendados = pd.read_parquet('top3_recomendados.parquet')
top3_no_recomendados = pd.read_parquet('top3_no_recomendados.parquet')
developer_sentiment = pd.read_parquet('developer_sentiment.parquet')

# Usar dataframes desde parquet (recomendación)
unique_item_ids = pd.read_parquet('unique_item_ids.parquet')
user_name_time = pd.read_parquet('user_name_time.parquet')
piv_table_norm = pd.read_parquet('piv_table_norm.parquet')
df_item_simil = pd.read_parquet('df_item_simil.parquet')
df_user_simil = pd.read_parquet('df_user_simil.parquet')


# Se instancia la aplicación
app = FastAPI(title="PROYECTO INDIVIDUAL Nº1 - Machine Learning Operations (MLOps) - Amelia Herrera Briceño - melinnicri@gmail.com",
            description="API de datos y recomendaciones de juegos de video online STEAM")


# Función para reconocer el servidor local
@app.get("/")
async def index():
    return {"Hola! Bienvenido a la API de consulta y recomedación. Por favor dirígete a /docs"}

@app.get("/about/")
async def about():
    return {"PROYECTO INDIVIDUAL Nº1 -Machine Learning Operations (MLOps)"}


# Funciones
@app.get(path="/", 
         response_class=HTMLResponse,
         tags=["Home"])
def home():

    '''
    Consultas sobre videojuegos en línea Steam

    Returns:
    Ni me pregunten cómo la eché a andar!!
    '''
    return presentacion()


@app.get(path = "/playtimegenre",
          description = """ <font color="blue">
                        Año con más horas jugadas<br>
                        Devuelve año con mas horas jugadas para dicho género.
                        </font>
                        """,
         tags=["Consultas"])
def PlayTimeGenre(genero: str = Query(..., 
                                description="Genero de videojuego", 
                                example="Massively Multiplayer")):
        
    return PlayTimeGenre(genero)

@app.get(path = "/userforgenre",
          description = """ <font color="blue">
                        Usuario con más horas jugadas<br>
                        Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
                        </font>
                        """,
         tags=["Consultas"])
def UserForGenre(genero: str = Query(..., 
                                description="Genero de videojuego", 
                                example="Action")):
    return UserForGenre(genero)

@app.get(path = "/usersrecommend",
          description = """ <font color="blue">
                        MAS recomendados<br>
                        Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.
                        </font>
                        """,
         tags=["Consultas"])
def UsersRecommend(anio: int = Query(..., 
                                description="Año a consultar.", 
                                example="2014")):
    return UsersRecommend(anio)


@app.get(path = "/usersworstdeveloper",
          description = """ <font color="blue">
                        MENOS recomendados<br>
                        Devuelve el top 3 de desarrolladores MENOS recomendados por usuarios para el año dado.
                        </font>
                        """,
         tags=["Consultas"])
def UsersWorstDeveloper(anio: int = Query(..., 
                                description="Año a consultar.", 
                                example="2014")):
    return UsersWorstDeveloper(anio)

@app.get(path = "/sentimentanalysis",
          description = """ <font color="blue">
                        Desarrollador<br>
                        Devuelve lista de registros totales de reseñas Negativa, Positiva, o Neutral de los Usuarios.
                        </font>
                        """,
         tags=["Consultas"])
def SentimentAnalysis(desarrollador: str = Query(..., 
                                description="Desarrollador para obtener conteo de reseñas", 
                                example="Studio SiestA")):
    return SentimentAnalysis(desarrollador)


@app.get(path = "/recomendacionjuego",
          description = """ <font color="blue">
                        Recomendación<br>
                        Devuelve lista de 5 juegos similares al nombre del Juego ingresado.
                        </font>
                        """,
         tags=["Consultas"])
def recommended_games_item(game, df_item_simil: str = Query(..., 
                                description="Recomendación de 5 juegos similares al nombre del Juego ingresado", 
                                example="the static speaks my name")):
    return recommended_games_item(game, df_item_simil)


@app.get(path = "/recomendacionjuegousuario",
          description = """ <font color="blue">
                        Recomendación<br>
                        Devuelve lista de 5 juegos similares según el Usuario ingresado.
                        </font>
                        """,
         tags=["Consultas"])
def similar_user_recs(user: str = Query(..., 
                                description="Recomendación de 5 juegos similares según el Usuario ingresado", 
                                example="virtueavatar")):
    return similar_user_recs(user)
