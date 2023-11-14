# app.py
import os
import pandas as pd
import numpy as np
from konlpy.tag import Okt
from gensim.models.word2vec import Word2Vec
import re
import html
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from fastapi import FastAPI

app = FastAPI()

# Set Java environment variable
os.environ["JAVA_HOME"] = "C:\Program Files\Java\jdk-11"

file = "https://drive.google.com/uc?id=1wodCdEkTU1HKztALyhqcuSf2tm0g1Y5c"

# Load musical data from CSV
musical_data = pd.read_csv(file, encoding='utf-8')

# Clean special characters and HTML entities from all columns
for col in musical_data.columns:
    musical_data[col] = musical_data[col].apply(lambda x: re.sub(r'[^\w\s]', '', html.unescape(str(x))))

# Text preprocessing using Konlpy
twitter = Okt()

def preprocessingText(synopsis):
    stems = []
    tagged_review = twitter.pos(synopsis, stem=True)
    
    for word, pos in tagged_review:
        if pos == "Noun" or pos == 'Adjective':
            stems.append(word)
    
    return " ".join(stems)

musical_data['synopsis_clear'] = musical_data['synopsis'].fillna('').apply(preprocessingText)

# Word2Vec model training
sentences = musical_data['synopsis'].tolist()
tokenized_data = musical_data['synopsis'].apply(lambda x: preprocessingText(str(x))).fillna('')
musical_data["synopsis_clear"] = musical_data['synopsis_clear'].astype(str) + " "
musical_data["tokenized_data"] = musical_data["synopsis_clear"].apply(lambda data: data.split(" "))

model = Word2Vec(musical_data["tokenized_data"],
                 vector_size=100,
                 window=3,
                 min_count=2,
                 sg=1)

model.save("word2vec_model2.bin")

# Convert WordVector words to string
word2vec_words = model.wv.key_to_index.keys()

# Vectorize synopsis and store as strings
string_array = []

for index in range(len(musical_data)):
    NUM = musical_data.loc[index, "musical_id"]
    TITLE = musical_data.loc[index, "title"]
    LINE = musical_data.loc[index, "tokenized_data"]

    doc2vec = None
    count = 0

    for word in LINE:
        if word in word2vec_words:
            count += 1
            if doc2vec is None:
                doc2vec = model.wv[word]
            else:
                doc2vec = doc2vec + model.wv[word]

    if doc2vec is not None:
        doc2vec = doc2vec / count

    string_array.append(doc2vec.tostring())

musical_data["doc2vec_vec"] = string_array

# Extract start_date and end_date from date
musical_data['start_date'] = pd.to_datetime(musical_data['date'].str[:8], format="%Y%m%d")
musical_data['end_date'] = pd.to_datetime(musical_data['date'].str[:6], format="%Y%m")

# Reorder columns
musical_data = musical_data[['musical_id', 'title', 'poster_url', 'genre', 'date', 'start_date', 'end_date', 'location',
                             'actors', 'age_rating', 'running_time', 'describe', 'synopsis', 'synopsis_clear',
                             'tokenized_data', 'doc2vec_vec']]

# Save to CSV if needed
# musical_data.to_csv('../musical_data_vector.csv', index=False, encoding='utf-8')

# Recommender System
# Convert doc2vec_vec to numpy array
musical_data["doc2vec_numpy"] = musical_data["doc2vec_vec"].apply(lambda x: np.fromstring(x, dtype="float32"))

# Scale the data
scaler = StandardScaler()
scaler.fit(np.array(musical_data["doc2vec_numpy"].tolist()))
musical_data["doc2vec_numpy_scale"] = scaler.transform(np.array(musical_data["doc2vec_numpy"].tolist())).tolist()

# Calculate Euclidean distances
sim_score = euclidean_distances(musical_data["doc2vec_numpy_scale"].tolist(), musical_data["doc2vec_numpy_scale"].tolist())
sim_df = pd.DataFrame(data=sim_score, index=musical_data["title"], columns=musical_data["title"])

# Example: Get 5 musicals similar to "빨간모자2"
similar_musicals = sim_df["빨간모자2"].sort_values()[1:6]
print(similar_musicals)

@app.get("/")
def init():
    return similar_musicals
