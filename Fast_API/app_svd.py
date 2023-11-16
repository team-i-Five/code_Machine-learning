# 필요한 라이브러리 및 모듈을 임포트
from fastapi import FastAPI # # FastAPI는 웹 API를 빠르게 작성할 수 있도록 도와주는 프레임워크 -> 웹서버로 데이터가 잘 전달 되는지 확인할 수 있다.
import pandas as pd # pandas는 데이터 조작과 분석을 위한 라이브러리
import numpy as np  # numpy는 수치 계산을 위한 라이브러리
# TruncatedSVD, cosine_similarity, StandardScaler는 scikit-learn 라이브러리의 일부로, 각각 특이값 분해, 코사인 유사도 계산, 데이터 스케일링 등을 위해 사용
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pymysql # pymysql은 MySQL 데이터베이스와의 연결을 위한 라이브러리
import ast # ast는 문자열을 파이썬 객체로 변환하는데 사용
from fastapi.responses import JSONResponse # JSONResponse는 FastAPI에서 응답을 생성하기 위한 클래스

# FastAPI 인스턴스 생성
app_svd = FastAPI() 

# MySQL 데이터베이스에서 데이터를 로드하는 함수 정의
def load_data(sql):
    # MySQL 연결 설정
    db = pymysql.connect(
        host='ifive-db.ckteh9hwnkjf.ap-northeast-2.rds.amazonaws.com',
        port=3306,
        user='admin',
        passwd='ifive1234',
        db='ifive',
        charset='utf8'
    )
     # SQL 쿼리 실행 및 데이터프레임으로 변환
    df = pd.read_sql(sql, db)
    # MySQL 연결 닫기
    db.close()
    return df

# MySQL 데이터베이스에서 과거 및 현재 데이터를 로드
past_sql = "SELECT * FROM musical_past"
present_sql = "SELECT * FROM musical_present"

# 데이터 로드
past_data = load_data(past_sql)
present_data = load_data(present_sql)

# SVD 및 유사도 계산
@app_svd.get("/recommend/{musical_id}")
def recommend(musical_id: int):
    try:
        # 과거 작품 선택
        # "past_data['musical_id'] == musical_id"는 'musical_id'가 주어진 값과 일치하는 행을 선택
        # 그 후, 해당 행의 인덱스 중 첫 번째 값을 선택 -> 즉, 'musical_id'가 'musical_id' 값과 일치하는 첫 번째 행의 인덱스를 선택하는 것
        selected_work_index_past = past_data[past_data['musical_id'] == musical_id].index[0]

        # 과거 데이터 파싱 및 스케일링
        # "ast.literal_eval"은 문자열을 파이썬 리터럴 구문으로 평가 -> 문자열을 리스트로 변환
        # 'x.decode('utf-8')'은 바이트열을 문자열로 디코딩 -> 데이터베이스에서 가져온 값이 바이트열로 저장되어 있을 수 있기 때문
        past_data['synopsis_numpy_scale'] = past_data['synopsis_numpy_scale'].apply(lambda x: ast.literal_eval(x.decode('utf-8')))
        scaler_past = StandardScaler()
        # 데이터의 스케일을 조정하기 위해 "StandardScaler" 객체를 생성
        # 이 객체는 각 특성의 평균을 0, 표준 편차를 1로 만들어주는 스케일 조정을 수행
        past_data_scaled = scaler_past.fit_transform(np.vstack(past_data['synopsis_numpy_scale']))
        # "np.vstack"은 수직으로 배열을 쌓아주는 함수 -> 'synopsis_numpy_scale' 열의 값을 수직으로 쌓아 2D 배열로 만듬
        # "fit_transform" 메서드를 사용하여 데이터를 스케일 조정 -> 이를 통해 모든 특성의 스케일이 조정된 데이터 얻음
        svd_past = TruncatedSVD(n_components=10)
        # "TruncatedSVD"는 특잇값 분해(SVD)를 수행하는 알고리즘 중 하나 -> 주어진 차원(여기서는 10)으로 데이터를 압축
        transformed_past_data = svd_past.fit_transform(past_data_scaled) 
        # "fit_transform" 메서드를 사용하여 스케일이 조정된 데이터에 대해 특잇값 분해를 수행 -> 차원이 감소된 데이터를 얻을 수 있음

        # 현재 데이터 파싱 및 스케일링
        present_data['synopsis_numpy_scale'] = present_data['synopsis_numpy_scale'].apply(lambda x: ast.literal_eval(x.decode('utf-8')))
        scaler_present = StandardScaler()
        present_data_scaled = scaler_present.fit_transform(np.vstack(present_data['synopsis_numpy_scale']))
        svd_present = TruncatedSVD(n_components=10)
        transformed_present_data = svd_present.fit_transform(present_data_scaled)

        # 코사인 유사도 계산
        selected_work = transformed_past_data[selected_work_index_past].reshape(1, -1)
        # "selected_work"는 선정된(현재 요청된) 작품의 SVD 변환된 데이터를 나타냄
        # SVD 변환 결과는 여러 차원으로 구성되어 있으며, 여기서는 1차원 배열로 변형
        similarities = cosine_similarity(transformed_present_data, selected_work)
        # "cosine_similarity" 함수를 사용하여 현재 데이터셋에 있는 모든 작품과
        # 선정된 작품 간의 코사인 유사도를 계산
        # 이는 두 작품 간의 유사도를 나타내며, 값이 높을수록 두 작품은 서로 유사

        # 정렬하여 상위 유사 작품 출력
        # "argsort" 함수는 배열의 각 요소를 정렬하는 데 사용
        # 여기서는 현재 데이터셋에 있는 모든 작품들과의 유사도를 기준으로 정렬
        # "axis=0"은 열 방향으로 정렬하라는 의미이며, 각 작품에 대한 정렬 결과를 반환
        # "[::-1]"은 역순으로 정렬하라는 의미이므로, 유사도가 높은 순서대로 정렬
        # "flatten()"은 2D 배열을 1D 배열로 평탄화
        similar_work_indices = similarities.argsort(axis=0)[::-1].flatten()
        top_n = 5

        result = [] # 추천 사항을 저장할 빈 리스트 생성
        for i in range(1, top_n + 1):
            index = similar_work_indices[i] # i번째 유사 작품의 인덱스 가져오기
            similarity = float(similarities[index]) # i번째 유사 작품의 유사도 점수 가져오기
            title = present_data.loc[index, 'title'] # i번째 유사 작품의 제목 가져오기
            musical_id = int(present_data.loc[index, 'musical_id']) # i번째 유사 작품의 musical_id 가져오기
            result.append({"title": title, "musical_id": musical_id, "similarity": similarity}) # 제목, musical_id 및 유사도를 포함하는 딕셔너리를 result 리스트에 추가

        return result # 추천 목록을 반환

    except Exception as e:
        # 예외가 발생한 경우 실행되는 블록
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500) # JSONResponse를 사용하여 오류 메시지와 500 상태 코드를 클라이언트에게 반환
    
# 실행방법 : uvicorn app_svd:app_svd --reload --host 0.0.0.0 --port 8080
# 주소검색 : http://localhost:8080/recommend/{musical_id}