import json # JSON 데이터를 다루기 위한 라이브러리
import pymysql # MySQL 데이터베이스에 연결하고 쿼리 실행을 위한 라이브러리
import pandas as pd # 데이터프레임을 사용하기 위한 판다스 라이브러리
import numpy as np # 배열 및 행렬 연산을 위한 넘파이 라이브러리
from sklearn.neighbors import NearestNeighbors # NearestNeighbors는 주어진 데이터셋에 대해 k-최근접 이웃을 찾는 데 사용되는 클래스 -> 주어진 데이터포인트와 가장 가까운 이웃들을 찾을 수 있음
from sklearn.preprocessing import StandardScaler # 데이터 스케일링을 위한 스케일러 제공하는 라이브러리 -> 데이터의 각 특성을 평균이 0이고 표준편차가 1이 되도록 변환
from fastapi import FastAPI  # FastAPI 웹 프레임워크
from fastapi.responses import JSONResponse # FastAPI에서 사용할 JSON 응답 생성을 위한 라이브러리

# FastAPI 인스턴스 생성
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

# 뮤지컬 추천을 위한 musical_id에 기반한 엔드포인트 정의
@app_knn.get("/recommend/{musical_id}")
def recommend(musical_id: int):
    try:
        # 선택한 작품의 인덱스 찾기
        selected_work_index_past = past_data[past_data['musical_id'] == musical_id].index[0]
        
        # 데이터프레임에서 synopsis_numpy_scale 열의 값을 파싱하여(JSON으로 로드하여) 리스트로 변환
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
        # 최대 거리와 최소 거리를 이용하여 거리 값을 [0, 1] 범위로 정규화
        # 정규화된 유사도를 계산하여 유사도 값이 1에 가까울수록 더 유사한 항목
        # distances 배열에서 각 거리 값을 정규화
        normalized_distances = (1 - (distances - min_distance) / (max_distance - min_distance))

        if len(distances[0]) <= 1:  # 유사한 항목이 부족할 경우, 즉, 적어도 2개 이상의 데이터가 필요한 경우 에러 응답을 반환
            return JSONResponse(content={"error": "Not enough similar items found."}, status_code=500)
            # 클라이언트에게 충분한 유사한 항목을 찾을 수 없다는 내용의 에러 메시지를 전송하고 상태 코드 500을 반환

        # 유사도가 높은 순서대로 정렬하여 유사한 작품 인덱스를 찾기
        # argsort 함수를 사용하여 정렬된 인덱스를 얻고, [::-1]을 사용하여 역순으로 정렬
        # 이후 flatten 함수를 사용하여 1차원 배열로 변환
        similar_work_indices = normalized_distances.argsort(axis=0)[::-1].flatten()
        # 상위 N개의 유사한 작품을 선택하되, 실제 유사한 작품의 수를 벗어나지 않도록
        top_n = min(5, len(similar_work_indices))
        
        # 상위 N개의 유사한 작품에 대한 정보를 추출하고 결과 리스트에 추가
        # 결과 리스트에는 각 작품의 제목, musical_id, 그리고 정규화된 유사도(similarity)가 포함
        result = []  # 추천 사항을 저장할 빈 리스트 생성
        # 유사한 작품 출력
        for i in range(top_n):
            index = similar_work_indices[i]
            similarity = float(distances[0][index])  # NumPy float로 변환
            # 작품 정보 추출
            title = present_data.loc[index, 'title']
            musical_id = int(present_data.loc[index, 'musical_id'])  # NumPy int64를 Python int로 변환
            # 결과 리스트에 작품 정보를 추가
            result.append({"title": title, "musical_id": musical_id, "similarity": similarity})
        # 추천 목록을 반환
        return result 
    except Exception as e:
         # 예외가 발생한 경우, 에러 응답을 반환
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)

# 실행방법 : uvicorn app_knn:app_knn --reload --host 0.0.0.0 --port 8080
# 주소검색 : http://localhost:8080/recommend/{musical_id}