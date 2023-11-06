# code_Machine-learning
프로젝트 머신 러닝 코드 repo

##START
#### linux와 git이 연결되어있지 않는 경우
git clone git@github.com:team-i-Five/code_Machine-learning.git
cd code_Machine-learning
[폴더안에 .git이 있는지 확인]
없을 경우 git init 진행하기

#### 항상 변경사항이 있는지 확인하기 위해 
git pull 진행

##### 테스트를 진행할 branch 생성
git branch 버전명/브랜치명
git checkout 생성한 버전명/브랜치명
![branch_ex](images/branch_ex.png)
##### 가상환경 생성
mkdir ~/ML
python3 -m venv ~/ML
source ~/ML/bin/activate
![ml_ex1](images/ml_ex1.png)
![ml_ex2](images/ml_ex2.png)
![ml_ex3](images/ml_ex3.png)

##### main 코드 branch로 가져오기
git pull origin main 진행

#### vscode실행
code .