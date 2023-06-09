#https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
#https://www.kaggle.com/datasets/gazu468/tmdb-10000-movies-dataset


import pandas as pd
import matplotlib.pyplot as plt
import io
import json
from flask import Flask, request, render_template
from src.utils import CorrectTitle, save_object,load_object

# df = pd.read_csv('Misc/tmdb_10000_movies.csv')
# print(df.head())

application = Flask(__name__)

app = application
print(__name__)
df = load_object('artifacts/movie_imgs/250.sav')

@app.route('/')
def home():
    return render_template('index.html',title = "Home")


@app.route('/recommend')
def results():
    return render_template('recommender.jinja2',title = "Recommendations")

@app.route('/show')
def show():
    return render_template('recommender.jinja2',title = "Recommendations",content = df[192])

@app.route('/display')
def display():
    images = [{'movie':x[0]+f"({x[1]})",'link':x[2]} for _,x in df[:5]]
    return render_template('gallery.html',title = "Recommendations",image_names = images)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True) # map to http://127.0.0.1:5000/

