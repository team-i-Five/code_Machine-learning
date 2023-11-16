import os
import ast
import json
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI

app = FastAPI()

# 과거 데이터 불러오기
def load_past_data():
    past_file = "https://drive.google.com/uc?id=1auGwLk9Jd69DOGEAEwCoOnmq0hLMY_24"
    past_chunks_data = pd.read_csv(past_file, encoding='utf-8', chunksize=1000)
    past_data = pd.DataFrame()
    for chunk in past_chunks_data:
        past_data = pd.concat([past_data, chunk], ignore_index=True)

    # 'synopsis_numpy_scale' 열의 JSON 문자열을 파이썬 리스트로 변환
    try:
        past_data['synopsis_numpy_scale'] = past_data['synopsis_numpy_scale'].apply(json.loads)
    except KeyError as e:
        print("Error in loading past data:", e)
        print("Past Data Columns:", past_data.columns)
        # 'synopsis_numpy_scale' 열이 존재하는지 확인
        print("Columns in Past Data:", past_data.columns)
        print("'synopsis_numpy_scale' in Past Data:", 'synopsis_numpy_scale' in past_data.columns)
        raise

    scaler_past = StandardScaler()
    past_data_scaled = scaler_past.fit_transform(np.vstack(past_data['synopsis_numpy_scale']))

    svd_past = TruncatedSVD(n_components=10)
    transformed_past_data = svd_past.fit_transform(past_data_scaled)

    return past_data, transformed_past_data, scaler_past

# 현재 데이터 불러오기
def load_present_data():
    present_file = "https://drive.google.com/uc?id=162bCdM5WGxxjZcUxHOab6ff-SYGUWDl7"
    present_chunks_data = pd.read_csv(present_file, encoding='utf-8', chunksize=1000)
    present_data = pd.DataFrame()
    for chunk in present_chunks_data:
        present_data = pd.concat([present_data, chunk], ignore_index=True)

    # 'synopsis_numpy_scale' 열의 JSON 문자열을 파이썬 리스트로 변환
    try:
        present_data['synopsis_numpy_scale'] = present_data['synopsis_numpy_scale'].apply(json.loads)
    except KeyError as e:
        print("Error in loading present data:", e)
        print("Present Data Columns:", present_data.columns)
        print("Columns in Present Data:", present_data.columns)
        print("'synopsis_numpy_scale' in Present Data:", 'synopsis_numpy_scale' in present_data.columns)
        raise

    scaler_present = StandardScaler()
    present_data_scaled = scaler_present.fit_transform(np.vstack(present_data['synopsis_numpy_scale']))

    svd_present = TruncatedSVD(n_components=10)
    transformed_present_data = svd_present.fit_transform(present_data_scaled)

    return present_data, transformed_present_data, scaler_present

# API 엔드포인트 정의
@app.get("/get_similar_works/{musical_id}")
async def get_similar_works(musical_id: int):
    # 과거 데이터와 관련된 정보를 불러오기
    past_data, transformed_past_data = load_past_data()

    # 현재 데이터와 관련된 정보를 불러오기
    present_data, transformed_present_data = load_present_data()

    # 선택된 작품의 인덱스를 찾기
    selected_work_index_past = past_data[past_data['musical_id'] == musical_id].index[0]

    # 선택된 작품의 특징을 추출하고 형태를 변경
    selected_work = transformed_past_data[selected_work_index_past].reshape(1, -1)

    # 선택된 작품과 다른 작품들 간의 유사도 계산
    similarities = cosine_similarity(transformed_present_data, selected_work)

    # 유사도를 기준으로 정렬한 후 상위 N개의 작품 인덱스 얻기
    similar_work_indices = similarities.argsort(axis=0)[::-1].flatten()
    top_n = 5
    similar_works = []

    # 상위 N개의 작품에 대한 정보 수집
    for i in range(1, top_n + 1):
        index = similar_work_indices[i]
        similarity = similarities[index][0]
        title = present_data.loc[index, 'title']
        musical_id = present_data.loc[index, 'musical_id']
        similar_works.append({"title": title, "musical_id": musical_id, "similarity": similarity})

    # 결과 반환
    return {
        "selected_work": {
            "title": past_data.loc[selected_work_index_past, 'title'],
            "musical_id": musical_id
        },
        "similar_works": similar_works
    }
