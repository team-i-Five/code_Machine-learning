import json
import ast
import pymysql
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI 
from fastapi.responses import JSONResponse

app_knn = FastAPI()

# MySQL 연결 설정 (애플리케이션 시작 시에 연결을 열고 종료 시에 닫음)
db = pymysql.connect(
    host='ifive-db.ckteh9hwnkjf.ap-northeast-2.rds.amazonaws.com',
    port=3306,
    user='admin',
    passwd='ifive1234',
    db='ifive',
    charset='utf8'
)

# MySQL 데이터베이스에서 과거 및 현재 데이터를 로드
past_sql = "SELECT * FROM musical_past"
present_sql = "SELECT * FROM musical_present"

# 데이터 로드
past_data = pd.read_sql(past_sql, db)
present_data = pd.read_sql(present_sql, db)

# MySQL 연결 닫기 (애플리케이션이 종료될 때)
db.close()

@app_knn.get("/recommend/{musical_id}")
def recommend(musical_id: int):
    try:
        # 과거 작품 선택
        selected_work_index_past = past_data[past_data['musical_id'] == musical_id].index[0]
        
        # 데이터프레임에서 synopsis_numpy_scale 열의 값을 파싱하여 리스트로 변환
        past_data['synopsis_numpy_scale'] = past_data['synopsis_numpy_scale'].apply(json.loads)
        # StandardScaler를 사용하여 특성들을 표준 스케일링
        scaler_past = StandardScaler()
        past_data_scaled = scaler_past.fit_transform(np.vstack(past_data['synopsis_numpy_scale']))
        
        # KNN 모델 초기화
        knn_model_past = NearestNeighbors(n_neighbors=7, metric='euclidean')
        knn_model_past.fit(past_data_scaled)
        
        # 현재 작품 선택
        # 데이터프레임에서 synopsis_numpy_scale 열의 값을 파싱하여 리스트로 변환
        present_data['synopsis_numpy_scale'] = present_data['synopsis_numpy_scale'].apply(json.loads)
        # StandardScaler를 사용하여 특성들을 표준 스케일링
        scaler_present = StandardScaler()
        present_data_scaled = scaler_present.fit_transform(np.vstack(present_data['synopsis_numpy_scale']))
        
        # KNN 모델 초기화
        knn_model_present = NearestNeighbors(n_neighbors=6, metric='euclidean')
        knn_model_present.fit(present_data_scaled)
        
        # 선택한 작품과 유사한 작품 찾기
        distances, indices = knn_model_present.kneighbors([present_data_scaled[selected_work_index_past]]) # 내적으로 유클리디안 거리가 계산됨

        # 최대 거리와 최소 거리 계산
        max_distance = distances.max()
        min_distance = distances.min()

        # 정규화된 유사도 계산
        normalized_distances = (1 - (distances - min_distance) / (max_distance - min_distance))

        if len(distances[0]) <= 1:  # 적어도 2개 이상의 데이터가 필요
            return JSONResponse(content={"error": "Not enough similar items found."}, status_code=500)

        # 유사도가 높은 순서대로 정렬하여 유사한 작품 인덱스를 찾습니다.
        similar_work_indices = normalized_distances.argsort(axis=0)[::-1].flatten()
        top_n = min(5, len(similar_work_indices))  # Ensure we don't exceed the number of available similar items
        
        result = []  # 추천 사항을 저장할 빈 리스트 생성
        # 유사한 작품 출력
        # 유사한 작품 출력
        for i in range(top_n):
            index = similar_work_indices[i]
            similarity = float(distances[0][index])  # NumPy float로 변환
            title = present_data.loc[index, 'title']
            musical_id = int(present_data.loc[index, 'musical_id'])  # NumPy int64를 Python int로 변환
            result.append({"title": title, "musical_id": musical_id, "similarity": similarity})
        return result  # 추천 목록을 반환
    except Exception as e:
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)

# 실행방법 : uvicorn app_knn:app_knn --reload --host 0.0.0.0 --port 8080
# 주소검색 : http://localhost:8080/recommend/{musical_id}