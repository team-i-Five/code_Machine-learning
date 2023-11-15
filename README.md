# code_Machine-learning
프로젝트 머신러닝 코드 repo

## 추천 과정
**1. 사용자가 tag 3개를 선택하면 전체 데이터(train)에서 필터링된 데이터 보여줌(20개)**
→ 50개 이상이면 최신순으로 20개 보여줌

**2. 필터링된 50개의 데이터(train) 중에 사용자가 작품을 선택하면 해당 작품과 유사도가 높은 현재 상영중인 작품(present) 추천**
→ 즉, train데이터와 present데이터 줄거리 간의 유사도를 측정해서 추천

----
## 파일명
### DATA
- **train_data**\
`Data0_train.csv` : 원본 학습 데이터\
`Data1_cleaned_data.csv` : train.csv의 특수문자 및 HTML 엔터티 코드 제거한 데이터\
`Data2_past_vector.csv` : 날짜 처리 및 시놉시스 벡터 컬럼 추가된 데이터\
`Data3_vector_tag.csv` : Data2_past_vector.csv에 tag 3개 붙인 데이터

### DATA_PREPROCESSING
- **train_data_preprocessing**\
`train_synopsis_vector.ipynb` : 날짜 처리 및 시놉시스 벡터화 코드\
`train_tag.ipynb` : tag 3개 붙이는 코드


### MODEL
`content_musical.ipynb` : 컨텐츠 기반 추천

----
## branch
`0.1.1`(민정) → train 데이터 벡터화, tag \
`0.1.2`(수빈) → present 데이터 벡터화, tag \
`0.1.3`(민정) → model_test
