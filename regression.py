import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
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

    # 회귀용 타겟: 1거래일 뒤 로그수익률과 5거래일 뒤 누적 로그수익률
    # 분류용 타겟: 1거래일 뒤 로그수익률의 상승/하락 방향
    df['Target_Return_1d'] = df.groupby('Name')['Return'].shift(-1)
    df['Target_Return_5d'] = df.groupby('Name')['close'].transform(lambda x: np.log(x.shift(-5) / x))
    df = df.dropna()
    df['Target_Direction'] = (df['Target_Return_1d'] > 0).astype(int)

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
    X_test = test_df[feature_cols]
    y_train_cls = train_df['Target_Direction']
    y_test_cls = test_df['Target_Direction']

    # 회귀 타겟 분리
    y_train_reg_1d = train_df['Target_Return_1d']
    y_test_reg_1d = test_df['Target_Return_1d']
    y_train_reg_5d = train_df['Target_Return_5d']
    y_test_reg_5d = test_df['Target_Return_5d']

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("--- Preprocessing Execution Finished ---")


    # 튜닝 자동화를 위해 plt.show() 시각화 코드는 파이프라인에서 제외하거나 주석 처리합니다.
    return X_train_scaled, X_test_scaled, y_train_cls, y_test_cls, y_train_reg_1d, y_test_reg_1d, y_train_reg_5d, y_test_reg_5d


# ====================================================
# [Step 2] 분류 및 회귀 모델링
# ====================================================
def train_and_evaluate(X_train, X_test, y_train, y_test, model_type, hyperparams):
    # 회귀, 분류 모델링 코드
    if model_type == "classification":
        return train_classification_models(X_train, X_test, y_train, y_test, hyperparams)
    return train_regression_models(X_train, X_test, y_train, y_test, hyperparams)


def train_classification_models(X_train, X_test, y_train, y_test, hyperparams):
    results = {}
    return results


def train_regression_models(X_train, X_test, y_train, y_test, hyperparams):
    y_train_array = np.asarray(y_train)
    y_test_array = np.asarray(y_test)

    # 베이스라인 (수익률을 0 또는 전체 평균으로 예상)
    zero_pred = np.zeros_like(y_test_array, dtype=float)
    mean_pred = np.full_like(y_test_array, y_train_array.mean(), dtype=float)

    results = [
        evaluate_regression_predictions("Zero Baseline", y_test_array, zero_pred),
        evaluate_regression_predictions("Mean Baseline", y_test_array, mean_pred),
    ]

    # HGB 후보 조합
    hist_gb_param_grid = [
        {"max_iter": 200, "learning_rate": 0.03, "max_leaf_nodes": 31},
        {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15},
    ]

    # 후보별 학습 및 평가
    for params in hist_gb_param_grid:
        hist_gb_model = HistGradientBoostingRegressor(
            **params,
            random_state=42
        )
        hist_gb_model.fit(X_train, y_train)
        hist_gb_pred = hist_gb_model.predict(X_test)

        hist_gb_result = evaluate_regression_predictions(
            "Hist Gradient Boosting",
            y_test_array,
            hist_gb_pred
        )
        hist_gb_result.update(params)
        results.append(hist_gb_result)

    return pd.DataFrame(results)


def evaluate_regression_predictions(model_name, y_true, y_pred):
    # 회귀 평가 지표
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Direction_Accuracy": np.mean((y_pred > 0) == (y_true > 0)),
    }


def plot_regression_results(regression_results):
    # 회귀 결과 시각화
    import matplotlib.pyplot as plt

    plot_df = regression_results.copy()

    def make_model_label(row):
        if row["Model"] == "Hist Gradient Boosting":
            return (
                "HGB\n"
                f"iter={int(row['max_iter'])}, "
                f"lr={row['learning_rate']}, "
                f"leaf={int(row['max_leaf_nodes'])}"
            )
        return row["Model"].replace(" Baseline", "\nBaseline")

    plot_df["Model_Label"] = plot_df.apply(make_model_label, axis=1)

    metrics = ["MAE", "RMSE", "Direction_Accuracy"]
    titles = ["MAE", "RMSE", "Direction Accuracy"]
    horizons = [h for h in ["1d", "5d"] if h in plot_df["Target_Horizon"].unique()]
    model_order = plot_df["Model_Label"].drop_duplicates().tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for ax, metric, title in zip(axes, metrics, titles):
        pivot_df = (
            plot_df
            .pivot(index="Model_Label", columns="Target_Horizon", values=metric)
            .reindex(model_order)
        )

        x = np.arange(len(model_order))
        width = 0.8 / len(horizons)

        for idx, horizon in enumerate(horizons):
            offset = (idx - (len(horizons) - 1) / 2) * width
            ax.bar(x + offset, pivot_df[horizon], width=width, label=horizon)

        if metric == "Direction_Accuracy":
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(model_order, rotation=20, ha="right")
        ax.legend(title="Target")

    plt.show()
    return fig, axes


# ====================================================
# [Step 3] 최상위 파이프라인 통합 함수
# ====================================================
def run_data_science_pipeline(data_path, hyperparams=None):
    print("========== S&P 500 Pipeline Start ==========")

    # 1. 전처리 실행 (결과물인 X, y 데이터를 받아옴)
    X_train, X_test, y_train_cls, y_test_cls, y_train_reg_1d, y_test_reg_1d, y_train_reg_5d, y_test_reg_5d = preprocess_data(data_path)

    # 2. 모델 학습 및 평가 (받아온 데이터를 모델에 넣음)
    classification_results = train_and_evaluate(
        X_train, X_test, y_train_cls, y_test_cls, "classification", hyperparams
    )
    regression_1d_results = train_and_evaluate(
        X_train, X_test, y_train_reg_1d, y_test_reg_1d, "regression", hyperparams
    )
    # 타겟 기간 표시
    regression_1d_results["Target_Horizon"] = "1d"

    regression_5d_results = train_and_evaluate(
        X_train, X_test, y_train_reg_5d, y_test_reg_5d, "regression", hyperparams
    )
    regression_5d_results["Target_Horizon"] = "5d"

    regression_results = pd.concat([regression_1d_results, regression_5d_results], ignore_index=True)
    results = {
        "classification": classification_results,
        "regression": regression_results,
    }

    print("========== S&P 500 Pipeline End ==========")
    return results


if __name__ == "__main__":
    # 실행 테스트
    pipeline_results = run_data_science_pipeline("data/all_stocks_5yr.csv")
    print(pipeline_results["regression"])
    plot_regression_results(pipeline_results["regression"])
