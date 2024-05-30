#Python Imports 

import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#importing Data set

music = pd.read_csv("spotify_millsongdata.csv")
music = music.sample(5000).drop('link',axis=1).reset_index(drop=True)

#text cleaning

music['text'] = music['text'].str.lower().replace(r'^\w\s',' ').replace(r'^\n',' ',regex=True)
music.isnull().sum()

#Tokenization and streamer

stm = PorterStemmer()
def token(txt) :

    token = nltk.word_tokenize(txt)
    a = [stm.stem(i) for i in token]

    return " ".join(a)

#cosine similarity

music['text'].apply(lambda x: token(x))
tfid = TfidfVectorizer(analyzer = 'word',stop_words = 'english')
matrix = tfid.fit_transform(music['text'])
similary = cosine_similarity(matrix)

#recomender function

def recommender(song_name) :
    idx = music[music['song']==song_name].index[0]
    distance = sorted(list(enumerate(similary[idx])),reverse = True , key = lambda x: x[1])
    song = []
    for s_id in distance[1:5] :
        song.append(music.iloc[s_id[0]].song)
    return song

print(recommender("Burning My Bridges"))
