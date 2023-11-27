import json
import pymysql
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from functools import lru_cache  # lru_cache 사용

app_nmf = FastAPI()
background_tasks = BackgroundTasks()

# MySQL 연결 설정 (애플리케이션 시작 시에 연결을 열고 종료 시에 닫음)
db = pymysql.connect(
    host='ifive-db.ckteh9hwnkjf.ap-northeast-2.rds.amazonaws.com',
    port=3306,
    user='admin',
    passwd='ifive1234',
    db='ifive',
    charset='utf8'
)

# MySQL 데이터베이스에서 과거, 현재, 미래 데이터를 로드
past_sql = "SELECT * FROM musical_past"
present_sql = "SELECT * FROM musical_present"
future_sql = "SELECT * FROM musical_future"

# 데이터 로드
past_data = pd.read_sql(past_sql, db)
present_data = pd.read_sql(present_sql, db)
future_data = pd.read_sql(future_sql, db)

# MySQL 연결 닫기 (애플리케이션이 종료될 때)
db.close()

# NMF 모델 초기화
nmf = NMF(n_components=10, init='random', random_state=42)

# 과거 데이터 스케일링
past_data['synopsis_numpy_scale'] = past_data['synopsis_numpy_scale'].apply(lambda x: np.array(json.loads(x)))
scaler_past = StandardScaler()
past_data_scaled = scaler_past.fit_transform(np.vstack(past_data['synopsis_numpy_scale']))
past_data_scaled = past_data_scaled - np.min(past_data_scaled) + 1e-10

# 특성 행렬 생성
V_past = np.vstack(past_data_scaled)

# NMF 모델 훈련
W_past = nmf.fit_transform(V_past)


# lru_cache를 사용하여 결과를 캐싱하는 함수
@lru_cache()
def recommend_impl(musical_id: int):
    try:
        # 선택한 작품의 인덱스 찾기
        selected_work_index_past = past_data[past_data['musical_id'] == musical_id].index

        # 현재 작품 선택
        present_data['synopsis_numpy_scale'] = present_data['synopsis_numpy_scale'].apply(
            lambda x: np.array(json.loads(x.decode('utf-8'))))
        scaler_present = StandardScaler()
        present_data_scaled = scaler_present.fit_transform(np.vstack(present_data['synopsis_numpy_scale']))
        present_data_scaled = present_data_scaled - np.min(present_data_scaled) + 1e-10

        # 현재 상영중인 데이터에 대한 특성 행렬 생성
        V_present = np.vstack(present_data_scaled)

        # 미래 데이터 선택
        future_data['synopsis_numpy_scale'] = future_data['synopsis_numpy_scale'].apply(
            lambda x: np.array(json.loads(x.decode('utf-8'))))
        scaler_future = StandardScaler()
        future_data_scaled = scaler_future.fit_transform(np.vstack(future_data['synopsis_numpy_scale']))
        future_data_scaled = future_data_scaled - np.min(future_data_scaled) + 1e-10

        # 미래 데이터에 대한 특성 행렬 생성
        V_future = np.vstack(future_data_scaled)

        # NMF 모델을 사용하여 현재 상영중인 데이터의 특성 행렬 분해
        W_present = nmf.transform(V_present)

        # NMF 모델을 사용하여 미래 데이터의 특성 행렬 분해
        W_future = nmf.transform(V_future)

        # 선택한 작품과 다른 작품 간의 코사인 유사도 계산 (현재 데이터)
        selected_work_present = W_present[selected_work_index_past].reshape(1, -1)
        similarities_present = cosine_similarity(W_present, selected_work_present)

        # 선택한 작품과 다른 작품 간의 코사인 유사도 계산 (과거 데이터)
        selected_work_past = W_past[selected_work_index_past].reshape(1, -1)
        similarities_past = cosine_similarity(W_past, selected_work_past)

        # 선택한 작품과 다른 작품 간의 코사인 유사도 계산 (미래 데이터)
        selected_work_future = W_future[selected_work_index_past].reshape(1, -1)
        similarities_future = cosine_similarity(W_future, selected_work_future)

        # 유사도가 높은 순서대로 정렬하여 유사한 작품 인덱스를 찾기 (현재 데이터)
        similar_work_indices_present = similarities_present.argsort(axis=0)[::-1].flatten()
        top_n_present = min(5, len(similar_work_indices_present))

        # 유사도가 높은 순서대로 정렬하여 유사한 작품 인덱스를 찾기 (과거 데이터)
        similar_work_indices_past = similarities_past.argsort(axis=0)[::-1].flatten()
        top_n_past = min(5, len(similar_work_indices_past))

        # 유사도가 높은 순서대로 정렬하여 유사한 작품 인덱스를 찾기 (미래 데이터)
        similar_work_indices_future = similarities_future.argsort(axis=0)[::-1].flatten()
        top_n_future = min(5, len(similar_work_indices_future))

        # 현재 데이터에 대한 추천 결과
        present_result = []
        for i in range(top_n_present):
            index = similar_work_indices_present[i]
            similarity = float(similarities_present[index])
            title = present_data.loc[index, 'title']
            musical_id = int(present_data.loc[index, 'musical_id'])
            present_result.append({"title": title, "musical_id": musical_id, "similarity": similarity})

        # 과거 데이터에 대한 추천 결과
        past_result = []
        for i in range(top_n_past):
            index = similar_work_indices_past[i]
            similarity = float(similarities_past[index])
            title = past_data.loc[index, 'title']
            musical_id = int(past_data.loc[index, 'musical_id'])
            past_result.append({"title": title, "musical_id": musical_id, "similarity": similarity})

        # 미래 데이터에 대한 추천 결과
        future_result = []
        for i in range(top_n_future):
            index = similar_work_indices_future[i]
            similarity = float(similarities_future[index])
            title = future_data.loc[index, 'title']
            musical_id = int(future_data.loc[index, 'musical_id'])
            future_result.append({"title": title, "musical_id": musical_id, "similarity": similarity})

        return {"present": present_result, "past": past_result, "future": future_result}

    except Exception as e:
        # 예외가 발생한 경우, 에러 응답을 반환
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)


# FastAPI 엔드포인트에서 캐시를 사용하도록 수정
@app_nmf.get("/recommend/{musical_id}")
async def recommend(musical_id: int, background_tasks: BackgroundTasks):
    try:
        # 캐시된 결과 확인
        cached_result = recommend_impl(musical_id)

        if cached_result is None:
            # 결과가 캐시되어 있지 않으면 계산하고 캐시에 저장
            background_tasks.add_task(recommend_impl, musical_id)
            return {"message": "Task added to background. Please try again."}
        else:
            # 캐시된 결과 반환
            return {"result": cached_result}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


# 실행방법 : uvicorn app_nmf:app_nmf --reload --host 0.0.0.0 --port 8080
# 주소검색 : http://localhost:8080/recommend/{musical_id}
