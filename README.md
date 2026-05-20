# S&P 500 Stock Return Prediction 📈

## Project Overview

이 프로젝트는 **2026 데이터 사이언스 텀 프로젝트**의 일환으로 진행되는 S&P 500 주가 및 업종 데이터 분석 파이프라인입니다. 기업의 주가 수익률을 기반으로 한 회귀(Regression) 모델과 상승/하락을 예측하는 분류(Classification) 모델을 병행하여 구축합니다. 오픈소스 SW 기여를 목표로, 데이터 전처리부터 모델 학습 및 최적화(Top 5 조합 탐색)까지의 전 과정을 단일 파이프라인(Single top-level function)으로 통합하여 제공합니다.

## Dataset

* **Source:** [S&P 500 Stock Market Data (Kaggle)](https://www.kaggle.com/camnugent/sandp500)
* **Description:** S&P 500 기업들의 5년간 주가 기록(Open, High, Low, Close, Volume) 및 섹터(Sector) 정보.
* **Notice:** 원본 `.csv` 파일은 용량이 크므로 Git 저장소에 업로드하지 않습니다. 데이터 파일은 반드시 로컬 환경의 `data/` 디렉토리를 생성하여 보관해 주세요.

## Features & Pipeline

* **Data Preprocessing:** 결측치/이상치(Volume=0 또는 NaN) 정제, 수치형 변수 Scaling, 범주형 변수(Sector) Encoding.
* **Classification:** 주가의 상승/하락 및 급등 여부를 분류하며, 학습 시 반드시 **k-fold cross-validation**을 적용합니다.
* **Regression:** 내일의 수익률(%) 및 N일 누적 로그 수익률을 예측합니다.
* **Pipeline Integration:** `pipeline.py`의 최상위 함수를 통해 여러 하이퍼파라미터 조합을 탐색하고 최고의 성능을 낸 Top 5 조합을 반환합니다.

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/[본인아이디]/sp500-return-prediction.git
cd sp500-return-prediction

```

**2. Data Setup**
프로젝트 루트 경로에 `data` 폴더를 생성하고, Kaggle에서 다운로드한 `.csv` 파일을 해당 폴더 안에 위치시킵니다.

```text
sp500-return-prediction/
├── data/                  # 이 폴더를 직접 만들고 csv 파일을 넣어주세요 (.gitignore 처리됨)
│   └── all_stocks_5yr.csv
├── pipeline.py
├── README.md
└── .gitignore

```

**3. Run the Pipeline**

```bash
python pipeline.py

```

## Git Workflow (GitHub Flow)

우리 팀은 빠르고 충돌 없는 협업을 위해 **GitHub Flow**를 사용합니다.

* `main` 브랜치는 항상 실행 가능한 완벽한 상태를 유지합니다.
* 각자 맡은 기능이나 역할에 따라 새로운 브랜치를 생성하여 작업합니다. (예: `feat/preprocessing`, `feat/modeling-cls`)
* 작업이 완료되면 `main` 브랜치로 바로 Push하지 않고, Pull Request (PR)를 생성합니다.
* 단톡방에 PR 생성을 알리고, 통합 담당자의 리뷰 및 승인(Merge)을 거쳐 `main`에 반영합니다.

## Team Members & Roles

* **[팀원 1 이름]**: Data Preprocessing & EDA Lead
* **[팀원 2 이름]**: Classification Modeling Lead
* **[팀원 3 이름]**: Regression Modeling Lead
* **[통합 담당자 이름]**: Open Source Integration & GitHub Manager
* **[팀원 5 이름]**: Tech Writer & Presentation Lead
