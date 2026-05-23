import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings


# ====================================================
# [Step 1] 전처리 파이프라인 (유찬님 코드 통합)
# ====================================================
def preprocess_data(file_path):
    warnings.filterwarnings('ignore')
    print("--- [Step 1] Data Preprocessing Started ---")

    # 1. 하드코딩된 파일명 대신 함수의 인자(file_path)를 사용합니다.
    df = pd.read_csv(file_path)

    df = df.copy()

    # 결측치 및 0값 처리
    df['volume'] = df['volume'].replace(0, np.nan)
    df['close'] = df.groupby('Name')['close'].ffill()
    df['volume'] = df.groupby('Name')['volume'].transform(lambda x: x.fillna(x.mean()))

    # 피처 엔지니어링 (MACD, RSI, Return 등)
    df['Return'] = df.groupby('Name')['close'].transform(lambda x: np.log(x / x.shift(1)))
    df['MA_12'] = df.groupby('Name')['close'].transform(lambda x: x.rolling(window=12).mean())
    df['MA_26'] = df.groupby('Name')['close'].transform(lambda x: x.rolling(window=26).mean())
    df['MACD'] = df['MA_12'] - df['MA_26']

    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    df['RSI'] = df.groupby('Name')['close'].transform(lambda x: calculate_rsi(x))
    df['Target'] = df.groupby('Name')['Return'].shift(-1).apply(lambda x: 1 if x > 0 else 0)
    df = df.dropna()

    # 이상치 처리
    lower_bound = df['Return'].quantile(0.01)
    upper_bound = df['Return'].quantile(0.99)
    df['Return'] = np.clip(df['Return'], lower_bound, upper_bound)
    df['MACD'] = np.clip(df['MACD'], df['MACD'].quantile(0.01), df['MACD'].quantile(0.99))

    # 인코딩
    label_encoder = LabelEncoder()
    df['Name_Encoded'] = label_encoder.fit_transform(df['Name'])

    # 데이터 분할
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'] < '2017-01-01']
    test_df = df[df['date'] >= '2017-01-01']

    feature_cols = ['Return', 'volume', 'MACD', 'RSI', 'Name_Encoded']
    X_train = train_df[feature_cols]
    y_train = train_df['Target']
    X_test = test_df[feature_cols]
    y_test = test_df['Target']

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("--- Preprocessing Execution Finished ---")


    # 튜닝 자동화를 위해 plt.show() 시각화 코드는 파이프라인에서 제외하거나 주석 처리합니다.
    return X_train_scaled, X_test_scaled, y_train, y_test


# ====================================================
# [Step 2] 분류 및 회귀 모델링
# ====================================================
def train_and_evaluate(X_train, X_test, y_train, y_test, model_type, hyperparams):
    # 회귀, 분류 모델링 코드
    results = {}
    return results


# ====================================================
# [Step 3] 최상위 파이프라인 통합 함수
# ====================================================
def run_data_science_pipeline(data_path, model_type="classification", hyperparams=None):
    print("========== S&P 500 Pipeline Start ==========")

    # 1. 전처리 실행 (결과물인 X, y 데이터를 받아옴)
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    # 2. 모델 학습 및 평가 (받아온 데이터를 모델에 넣음)
    # results = train_and_evaluate(X_train, X_test, y_train, y_test, model_type, hyperparams)

    print("========== S&P 500 Pipeline End ==========")
    return  # results


if __name__ == "__main__":
    # 실행 테스트
    run_data_science_pipeline("data/all_stocks_5yr.csv")