import json # JSON 데이터를 다루기 위한 라이브러리
import pymysql # MySQL 데이터베이스에 연결하고 쿼리 실행을 위한 라이브러리
import pandas as pd # 데이터프레임을 사용하기 위한 판다스 라이브러리
import numpy as np # 배열 및 행렬 연산을 위한 넘파이 라이브러리
from sklearn.preprocessing import StandardScaler # 데이터 스케일링을 위한 스케일러 제공하는 라이브러리
from sklearn.decomposition import NMF # 비음수 행렬 분해(NMF) 모델을 사용하기 위한 라이브러리
from sklearn.metrics.pairwise import cosine_similarity # 코사인 유사도 계산을 위한 라이브러리
from fastapi import FastAPI # FastAPI 웹 프레임워크
from fastapi.responses import JSONResponse # FastAPI에서 사용할 JSON 응답 생성을 위한 라이브러리

# Fast API 생성
app_nmf = FastAPI()

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

# 뮤지컬 추천을 위한 musical_id에 기반한 엔드포인트 정의
@app_nmf.get("/recommend/{musical_id}")
def recommend(musical_id: int):
    try:
        # 선택한 작품의 인덱스 찾기
        selected_work_index_past = past_data[past_data['musical_id'] == musical_id].index[0]
        
        # 과거 데이터 파싱 및 스케일링 
        past_data['synopsis_numpy_scale'] = past_data['synopsis_numpy_scale'].apply(lambda x: np.array(json.loads(x.decode('utf-8'))))
        scaler_past = StandardScaler()
        past_data_scaled = scaler_past.fit_transform(np.vstack(past_data['synopsis_numpy_scale']))
        past_data_scaled = past_data_scaled - np.min(past_data_scaled) + 1e-10
        
        # 현재 작품 선택
        present_data['synopsis_numpy_scale'] = present_data['synopsis_numpy_scale'].apply(lambda x: np.array(json.loads(x.decode('utf-8'))))
        scaler_present = StandardScaler()
        present_data_scaled = scaler_present.fit_transform(np.vstack(present_data['synopsis_numpy_scale']))
        present_data_scaled = present_data_scaled - np.min(present_data_scaled) + 1e-10

        # NMF 모델 초기화
        # n_components: 추출할 특성의 수로, 작품을 나타내는 잠재적인 특징의 개수를 설정
        #  init: 행렬을 초기화하는 방법을 설정 -> 'random'은 무작위 초기화
        # random_state: 난수 발생을 제어하여 모델이 항상 일관된 결과를 생성하도록
        nmf = NMF(n_components=10, init='random', random_state=42)

        # 특성 행렬 생성
        # 행렬을 수직으로 쌓아서 새로운 행렬을 생성
        # 'past_data_scaled'는 각 작품의 특성을 행으로 가지고 있는 2D 배열
        # np.vstack 함수를 사용하여 이를 수직으로 쌓아서 특성 행렬 'V'를 생성
        V = np.vstack(past_data_scaled)

        # NMF 모델 훈련
        W = nmf.fit_transform(V) # W는 특성 행렬 -> W는 데이터를 특성으로 표현
        # H = nmf.components_ # H는 NMF 모델에서 생성된 행렬 중 하나로 주로 특성을 나타냄

        # 현재 상영중인 데이터에 대한 특성 행렬 생성
        V_present = np.vstack(present_data_scaled)

        # NMF 모델을 사용하여 현재 상영중인 데이터의 특성 행렬 분해
        W_present = nmf.transform(V_present)

        # 선택한 작품과 다른 작품 간의 코사인 유사도 계산
        selected_work = W[selected_work_index_past].reshape(1, -1)
        similarities = cosine_similarity(W_present, selected_work)

        # 유사도가 높은 순서대로 정렬하여 유사한 작품 인덱스를 찾기
        # argsort 함수를 사용하여 정렬된 인덱스를 얻고, [::-1]을 사용하여 역순으로 정렬
        # 이후 flatten 함수를 사용하여 1차원 배열로 변환
        similar_work_indices = similarities.argsort(axis=0)[::-1].flatten()
        # 상위 N개의 유사한 작품을 선택하되, 실제 유사한 작품의 수를 벗어나지 않도록
        top_n = min(5, len(similar_work_indices))
        
        # 상위 N개의 유사한 작품에 대한 정보를 추출하고 결과 리스트에 추가
        # 결과 리스트에는 각 작품의 제목, musical_id, 그리고 코사인 유사도(similarity)가 포함
        result = []  # 추천 사항을 저장할 빈 리스트 생성
        for i in range(top_n):
            index = similar_work_indices[i]
            similarity = float(similarities[index])  # NumPy float로 변환
            # 작품 정보 추출
            title = present_data.loc[index, 'title']
            musical_id = int(present_data.loc[index, 'musical_id'])  # NumPy int64를 Python int로 변환
             # 결과 리스트에 작품 정보를 추가
            result.append({"title": title, "musical_id": musical_id, "similarity": similarity})
        
        return result  # 추천 목록을 반환
    except Exception as e:
        # 예외가 발생한 경우, 에러 응답을 반환
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)
    
# 실행방법 : uvicorn app_nmf:app_nmf --reload --host 0.0.0.0 --port 8080
# 주소검색 : http://localhost:8080/recommend/{musical_id}