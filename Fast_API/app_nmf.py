import json
import pymysql
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

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

# 추가: 'synopsis_numpy_scale' 열의 데이터 타입 확인과 예시 값 출력
print("Data type of 'synopsis_numpy_scale' in past_data:", past_data['synopsis_numpy_scale'].dtype)
print("First few values of 'synopsis_numpy_scale' in past_data:", past_data['synopsis_numpy_scale'].head().tolist())

# 추가: 'synopsis_numpy_scale' 열의 데이터 타입 확인과 예시 값 출력
print("Data type of 'synopsis_numpy_scale' in present_data:", present_data['synopsis_numpy_scale'].dtype)
print("First few values of 'synopsis_numpy_scale' in present_data:", present_data['synopsis_numpy_scale'].head().tolist())

# 뮤지컬 추천을 위한 musical_id에 기반한 엔드포인트 정의
@app_nmf.get("/{musical_id}")
def recommend(musical_id: int):
    try:
        # 선택한 작품의 인덱스 찾기
        selected_work_index_past = past_data[past_data['musical_id'] == musical_id].index[0]

        # 과거 데이터 파싱 및 스케일링
        past_data['synopsis_numpy_scale'] = past_data['synopsis_numpy_scale'].apply(lambda x: np.array(json.loads(x.decode('utf-8')) if isinstance(x, bytes) else x))
        scaler_past = StandardScaler()
        past_data_scaled = scaler_past.fit_transform(np.vstack(past_data['synopsis_numpy_scale']))
        past_data_scaled = past_data_scaled - np.min(past_data_scaled) + 1e-10

        # 현재 작품 선택
        present_data['synopsis_numpy_scale'] = present_data['synopsis_numpy_scale'].apply(lambda x: np.array(json.loads(x.decode('utf-8')) if isinstance(x, bytes) else x))
        scaler_present = StandardScaler()
        present_data_scaled = scaler_present.fit_transform(np.vstack(present_data['synopsis_numpy_scale']))
        present_data_scaled = present_data_scaled - np.min(present_data_scaled) + 1e-10

        # 'list' 객체에 'tolist' 속성이 없는 경우를 고려하여 tolist 호출
        present_data['synopsis_numpy_scale'] = present_data['synopsis_numpy_scale'].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x if x is not None else [])

        # NMF 모델 초기화
        nmf = NMF(n_components=10, init='random', random_state=42, max_iter=500)

        # 특성 행렬 생성
        V = np.vstack(past_data_scaled)

        # NMF 모델 훈련
        W = nmf.fit_transform(V)

        # 현재 상영중인 데이터에 대한 특성 행렬 생성
        V_present = np.vstack(present_data_scaled)

        # NMF 모델을 사용하여 현재 상영중인 데이터의 특성 행렬 분해
        W_present = nmf.transform(V_present)

        # 선택한 작품과 다른 작품 간의 코사인 유사도 계산
        selected_work = W[selected_work_index_past].reshape(1, -1)
        similarities = cosine_similarity(W_present, selected_work)

        # 유사도가 높은 순서대로 정렬하여 유사한 작품 인덱스를 찾기
        similar_work_indices = similarities.argsort(axis=0)[::-1].flatten()
        top_n = min(5, len(similar_work_indices))

        # 상위 N개의 유사한 작품에 대한 정보를 추출하고 결과 리스트에 추가
        result = []
        for i in range(top_n):
            index = similar_work_indices[i]
            similarity = float(similarities[index])
            # 작품 정보 추출
            title = present_data.loc[index, 'title']
            musical_id = int(present_data.loc[index, 'musical_id'])
            # 결과 리스트에 작품 정보를 추가
            result.append({"title": title, "musical_id": musical_id, "similarity": similarity})

        return result
    except Exception as e:
        # 예외가 발생한 경우, 에러 응답을 반환
        return HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")