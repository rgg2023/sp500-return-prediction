"""
S&P 500 Stock Return Prediction Pipeline
=========================================
단일 최상위 함수(run_data_science_pipeline)를 통해
데이터 전처리 → 분류(k-Fold CV) → 회귀 → Top 5 조합 탐색까지
전 과정을 수행합니다.

Dataset: S&P 500 Stock Market Data (Kaggle)
https://www.kaggle.com/camnugent/sandp500
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
)


# ============================================================
# [Step 1] 데이터 전처리
# ============================================================

def preprocess_data(file_path):
    """
    CSV 파일을 읽어 결측치 처리, 피처 엔지니어링,
    이상치 제거, 인코딩, Train/Test 분할, 스케일링을 수행합니다.

    Parameters
    ----------
    file_path : str
        원본 CSV 파일 경로 (예: 'data/all_stocks_5yr.csv')

    Returns
    -------
    dict  키 목록:
        X_train_scaled, X_test_scaled  : np.ndarray  스케일링된 Feature 행렬
        y_train_cls,    y_test_cls     : pd.Series   분류 타겟 (0/1)
        y_train_1d,     y_test_1d      : pd.Series   회귀 타겟 – 1일 수익률
        y_train_5d,     y_test_5d      : pd.Series   회귀 타겟 – 5일 수익률
        feature_cols                   : list[str]   사용된 Feature 이름 목록
    """
    print("\n[Step 1] 데이터 전처리 시작 ...")

    df = pd.read_csv(file_path)
    df = df.copy()

    # ── 결측치 / 0값 처리 ──────────────────────────────────
    df['volume'] = df['volume'].replace(0, np.nan)
    df['close']  = df.groupby('Name')['close'].ffill()
    df['volume'] = df.groupby('Name')['volume'].transform(
        lambda x: x.fillna(x.mean())
    )

    # ── 피처 엔지니어링 ────────────────────────────────────
    # 로그 수익률
    df['Return'] = df.groupby('Name')['close'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    # MACD (12일 MA - 26일 MA)
    df['MA_12'] = df.groupby('Name')['close'].transform(
        lambda x: x.rolling(window=12).mean()
    )
    df['MA_26'] = df.groupby('Name')['close'].transform(
        lambda x: x.rolling(window=26).mean()
    )
    df['MACD'] = df['MA_12'] - df['MA_26']

    # RSI (14일)
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain  = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs    = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    df['RSI'] = df.groupby('Name')['close'].transform(calculate_rsi)

    # ── 타겟 생성 ──────────────────────────────────────────
    # 회귀 타겟: 1일 / 5일 뒤 로그수익률
    df['Target_Return_1d'] = df.groupby('Name')['Return'].shift(-1)
    df['Target_Return_5d'] = df.groupby('Name')['close'].transform(
        lambda x: np.log(x.shift(-5) / x)
    )
    df = df.dropna()

    # 분류 타겟: 1일 뒤 방향 (상승=1, 하락=0)
    df['Target_Direction'] = (df['Target_Return_1d'] > 0).astype(int)

    # ── 이상치 Clipping (1%~99%) ───────────────────────────
    for col in ['Return', 'MACD']:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lo, hi)

    # ── 범주형 인코딩 (Label Encoding) ────────────────────
    le = LabelEncoder()
    df['Name_Encoded'] = le.fit_transform(df['Name'])

    # ── Train / Test 분할 (시계열 기준) ───────────────────
    df['date'] = pd.to_datetime(df['date'])
    train_df = df[df['date'] < '2017-01-01']
    test_df  = df[df['date'] >= '2017-01-01']

    feature_cols = ['Return', 'volume', 'MACD', 'RSI', 'Name_Encoded']

    X_train = train_df[feature_cols]
    X_test  = test_df[feature_cols]

    # ── Feature Scaling (StandardScaler) ──────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # train 기준 fit
    X_test_scaled  = scaler.transform(X_test)         # test는 transform만

    print(f"  Train shape : {X_train.shape}")
    print(f"  Test  shape : {X_test.shape}")
    print("[Step 1] 전처리 완료.\n")

    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled' : X_test_scaled,
        'y_train_cls'   : train_df['Target_Direction'],
        'y_test_cls'    : test_df['Target_Direction'],
        'y_train_1d'    : train_df['Target_Return_1d'],
        'y_test_1d'     : test_df['Target_Return_1d'],
        'y_train_5d'    : train_df['Target_Return_5d'],
        'y_test_5d'     : test_df['Target_Return_5d'],
        'feature_cols'  : feature_cols,
    }


# ============================================================
# [Step 2] 분류 모델링 (Random Forest + k-Fold CV)
# ============================================================

def run_classification(data, k=5):
    """
    Random Forest Classifier를 k-Fold Cross Validation으로 학습·평가합니다.

    Parameters
    ----------
    data : dict   preprocess_data() 반환값
    k    : int    k-Fold 수 (기본 5)

    Returns
    -------
    dict  키 목록:
        cv_scores  : dict  k-Fold 평균 지표
        test_scores: dict  테스트셋 지표
        model      : 학습된 RandomForestClassifier
    """
    print("[Step 2] 분류 모델링 (Random Forest + {}-Fold CV) ...".format(k))

    X_train = data['X_train_scaled']
    X_test  = data['X_test_scaled']
    y_train = data['y_train_cls']
    y_test  = data['y_test_cls']

    # ── 모델 정의 ──────────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=50,   # 과적합 방지
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )

    # ── k-Fold 교차 검증 ───────────────────────────────────
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_validate(
        model, X_train, y_train,
        cv=skf,
        scoring=['accuracy', 'precision', 'recall', 'f1'],
        return_train_score=False,
    )

    cv_scores = {
        'accuracy' : scores['test_accuracy'].mean(),
        'precision': scores['test_precision'].mean(),
        'recall'   : scores['test_recall'].mean(),
        'f1'       : scores['test_f1'].mean(),
    }

    print(f"  [{k}-Fold CV 평균]  "
          f"Accuracy={cv_scores['accuracy']:.4f}  "
          f"F1={cv_scores['f1']:.4f}")

    # ── 테스트셋 최종 평가 ─────────────────────────────────
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_scores = {
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall'   : recall_score(y_test, y_pred),
        'f1'       : f1_score(y_test, y_pred),
    }

    print(f"  [Test Set]          "
          f"Accuracy={test_scores['accuracy']:.4f}  "
          f"F1={test_scores['f1']:.4f}")
    print("[Step 2] 분류 완료.\n")

    return {
        'cv_scores'  : cv_scores,
        'test_scores': test_scores,
        'model'      : model,
        'feature_importances': dict(zip(
            data['feature_cols'],
            model.feature_importances_
        )),
    }


# ============================================================
# [Step 3] 회귀 모델링
# ============================================================

def _eval_regression(name, y_true, y_pred):
    """회귀 평가 지표를 딕셔너리로 반환합니다."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        'Model'             : name,
        'MAE'               : mean_absolute_error(y_true, y_pred),
        'RMSE'              : np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2'                : r2_score(y_true, y_pred),
        'Direction_Accuracy': np.mean((y_pred > 0) == (y_true > 0)),
    }


def run_regression(data):
    """
    1일/5일 수익률 예측을 위한 회귀 모델을 학습·평가합니다.
    Baseline(Zero, Mean), Linear Regression, Ridge, HistGradientBoosting을 비교합니다.

    Parameters
    ----------
    data : dict   preprocess_data() 반환값

    Returns
    -------
    pd.DataFrame  각 모델·타겟 조합의 평가 지표
    """
    print("[Step 3] 회귀 모델링 ...")

    X_train = data['X_train_scaled']
    X_test  = data['X_test_scaled']

    # 회귀 모델 후보
    reg_models = {
        'LinearRegression': LinearRegression(),
        'Ridge(alpha=1)'  : Ridge(alpha=1.0),
        'Ridge(alpha=10)' : Ridge(alpha=10.0),
        'HGB(lr=0.03)'    : HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.03, max_leaf_nodes=31, random_state=42),
        'HGB(lr=0.05)'    : HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=42),
    }

    all_results = []

    for horizon, y_train, y_test in [
        ('1d', data['y_train_1d'], data['y_test_1d']),
        ('5d', data['y_train_5d'], data['y_test_5d']),
    ]:
        y_train_arr = np.asarray(y_train)
        y_test_arr  = np.asarray(y_test)

        # Baseline
        for bname, bpred in [
            ('Zero Baseline', np.zeros_like(y_test_arr, dtype=float)),
            ('Mean Baseline', np.full_like(y_test_arr, y_train_arr.mean(), dtype=float)),
        ]:
            row = _eval_regression(bname, y_test_arr, bpred)
            row['Target_Horizon'] = horizon
            all_results.append(row)

        # 학습 모델
        for mname, model in reg_models.items():
            model.fit(X_train, y_train_arr)
            pred = model.predict(X_test)
            row  = _eval_regression(mname, y_test_arr, pred)
            row['Target_Horizon'] = horizon
            all_results.append(row)

        print(f"  [{horizon} 타겟] 모델 {len(reg_models)}개 평가 완료.")

    print("[Step 3] 회귀 완료.\n")
    return pd.DataFrame(all_results)


# ============================================================
# [Step 4] Top 5 조합 탐색
# ============================================================

def find_top5_combinations(data, k=5):
    """
    다양한 전처리·모델·하이퍼파라미터 조합을 탐색하여
    분류(F1 기준) / 회귀-1d(RMSE 기준) Top 5를 반환합니다.

    Parameters
    ----------
    data : dict   preprocess_data() 반환값
    k    : int    분류 k-Fold 수

    Returns
    -------
    dict  키:
        'classification_top5' : pd.DataFrame
        'regression_1d_top5'  : pd.DataFrame
    """
    print("[Step 4] Top 5 조합 탐색 ...")

    X_train = data['X_train_scaled']
    X_test  = data['X_test_scaled']
    y_cls_train = data['y_train_cls']
    y_cls_test  = data['y_test_cls']
    y_reg_train = data['y_train_1d']
    y_reg_test  = data['y_test_1d']

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # ── 분류 후보 조합 ─────────────────────────────────────
    cls_candidates = [
        dict(n_estimators=100, max_depth=10,  min_samples_leaf=30,  class_weight='balanced'),
        dict(n_estimators=100, max_depth=20,  min_samples_leaf=50,  class_weight='balanced'),
        dict(n_estimators=200, max_depth=20,  min_samples_leaf=50,  class_weight='balanced'),
        dict(n_estimators=100, max_depth=30,  min_samples_leaf=100, class_weight='balanced'),
        dict(n_estimators=200, max_depth=30,  min_samples_leaf=100, class_weight='balanced'),
        dict(n_estimators=300, max_depth=20,  min_samples_leaf=50,  class_weight='balanced'),
        dict(n_estimators=100, max_depth=None,min_samples_leaf=50,  class_weight='balanced'),
        dict(n_estimators=100, max_depth=20,  min_samples_leaf=50,  class_weight=None),
    ]

    cls_results = []
    for params in cls_candidates:
        model = RandomForestClassifier(**params, n_jobs=-1, random_state=42)
        cv = cross_validate(
            model, X_train, y_cls_train,
            cv=skf, scoring=['accuracy', 'f1'], return_train_score=False,
        )
        model.fit(X_train, y_cls_train)
        y_pred = model.predict(X_test)
        row = {
            'n_estimators'    : params['n_estimators'],
            'max_depth'       : str(params['max_depth']),
            'min_samples_leaf': params['min_samples_leaf'],
            'class_weight'    : str(params['class_weight']),
            'CV_F1'           : cv['test_f1'].mean(),
            'CV_Accuracy'     : cv['test_accuracy'].mean(),
            'Test_F1'         : f1_score(y_cls_test, y_pred),
            'Test_Accuracy'   : accuracy_score(y_cls_test, y_pred),
        }
        cls_results.append(row)

    cls_df = (
        pd.DataFrame(cls_results)
        .sort_values('CV_F1', ascending=False)
        .reset_index(drop=True)
    )
    cls_top5 = cls_df.head(5)

    # ── 회귀 후보 조합 ─────────────────────────────────────
    reg_candidates = [
        ('LinearRegression', LinearRegression(), {}),
        ('Ridge(a=0.1)', Ridge(alpha=0.1), {}),
        ('Ridge(a=1)',   Ridge(alpha=1.0), {}),
        ('Ridge(a=10)',  Ridge(alpha=10.0), {}),
        ('HGB lr=0.01 leaf=31', HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.01, max_leaf_nodes=31, random_state=42), {}),
        ('HGB lr=0.03 leaf=31', HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.03, max_leaf_nodes=31, random_state=42), {}),
        ('HGB lr=0.05 leaf=15', HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=42), {}),
        ('HGB lr=0.05 leaf=31', HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=42), {}),
        ('HGB lr=0.1  leaf=31', HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.10, max_leaf_nodes=31, random_state=42), {}),
    ]

    y_train_arr = np.asarray(y_reg_train)
    y_test_arr  = np.asarray(y_reg_test)

    reg_results = []
    for name, model, _ in reg_candidates:
        model.fit(X_train, y_train_arr)
        pred = model.predict(X_test)
        row  = _eval_regression(name, y_test_arr, pred)
        reg_results.append(row)

    reg_df = (
        pd.DataFrame(reg_results)
        .sort_values('RMSE', ascending=True)   # RMSE 낮을수록 좋음
        .reset_index(drop=True)
    )
    reg_top5 = reg_df.head(5)

    print("  [분류 Top 5 – CV F1 기준]")
    print(cls_top5[['n_estimators','max_depth','min_samples_leaf','class_weight',
                     'CV_F1','Test_F1']].to_string(index=False))
    print()
    print("  [회귀 Top 5 – RMSE 기준 (1일 수익률)]")
    print(reg_top5[['Model','MAE','RMSE','R2','Direction_Accuracy']].to_string(index=False))
    print("\n[Step 4] Top 5 탐색 완료.\n")

    return {
        'classification_top5': cls_top5,
        'regression_1d_top5' : reg_top5,
    }


# ============================================================
# [Step 5] 시각화
# ============================================================

def _add_bar_labels(ax, bars, fmt='{:.4f}', fontsize=7, padding=0.001):
    """막대 위에 값 레이블을 추가합니다."""
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + padding,
            fmt.format(h),
            ha='center', va='bottom', fontsize=fontsize,
        )


def _add_hbar_labels(ax, bars, fmt='{:.5f}', fontsize=7, padding=0.0001):
    """가로 막대 오른쪽에 값 레이블을 추가합니다."""
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + padding,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(w),
            ha='left', va='center', fontsize=fontsize,
        )


def visualize_results(cls_result, reg_df, top5, feature_cols, save_path='pipeline_results.png'):
    """
    분류·회귀·Top 5 결과를 하나의 Figure로 시각화하고 저장합니다.
    y축 범위를 데이터에 맞게 좁혀 차이가 명확히 보이도록 개선합니다.
    """
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle('S&P 500 Pipeline Results', fontsize=16, fontweight='bold')

    # ── 1. 분류 CV vs Test 지표 비교 ──────────────────────
    ax1 = fig.add_subplot(3, 3, 1)
    metrics   = list(cls_result['cv_scores'].keys())
    cv_vals   = list(cls_result['cv_scores'].values())
    test_vals = [cls_result['test_scores'][m] for m in metrics]
    x = np.arange(len(metrics))
    b1 = ax1.bar(x - 0.2, cv_vals,   0.35, label='CV Mean',  color='steelblue', alpha=0.85)
    b2 = ax1.bar(x + 0.2, test_vals, 0.35, label='Test Set', color='tomato',    alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    # 차이가 보이도록 y축 범위 좁힘
    all_vals = cv_vals + test_vals
    ax1.set_ylim(min(all_vals) * 0.97, max(all_vals) * 1.06)
    _add_bar_labels(ax1, list(b1) + list(b2), fmt='{:.4f}', padding=0.001)
    ax1.set_title('Classification – CV vs Test')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.4)

    # ── 2. Feature Importance ─────────────────────────────
    ax2 = fig.add_subplot(3, 3, 2)
    fi = cls_result['feature_importances']
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    names, vals = zip(*fi_sorted)
    bars = ax2.barh(names, vals, color='mediumseagreen', alpha=0.85)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(vals) * 1.15)
    _add_hbar_labels(ax2, bars, fmt='{:.4f}', padding=0.002)
    ax2.set_title('Feature Importance (Random Forest)')
    ax2.set_xlabel('Importance Score')
    ax2.grid(axis='x', alpha=0.4)

    # ── 3. 회귀 1d RMSE – y축 범위 좁힘 ──────────────────
    ax3 = fig.add_subplot(3, 3, 3)
    reg_1d  = reg_df[reg_df['Target_Horizon'] == '1d'].copy()
    # Baseline 제외 모델만 색상 구분
    colors3 = ['#AAAAAA' if 'Baseline' in m else '#8B6BD6'
               for m in reg_1d['Model']]
    bars3 = ax3.bar(range(len(reg_1d)), reg_1d['RMSE'],
                    color=colors3, alpha=0.85)
    rmse_min = reg_1d['RMSE'].min()
    rmse_max = reg_1d['RMSE'].max()
    margin   = (rmse_max - rmse_min) * 3 if rmse_max > rmse_min else 0.0005
    ax3.set_ylim(rmse_min - margin, rmse_max + margin * 4)
    ax3.set_xticks(range(len(reg_1d)))
    ax3.set_xticklabels(reg_1d['Model'], rotation=35, ha='right', fontsize=7)
    _add_bar_labels(ax3, bars3, fmt='{:.5f}', padding=margin * 0.3)
    ax3.set_title('Regression RMSE (1-day)')
    ax3.set_ylabel('RMSE')
    ax3.grid(axis='y', alpha=0.4)

    # ── 4. 회귀 5d RMSE – y축 범위 좁힘 ──────────────────
    ax4 = fig.add_subplot(3, 3, 4)
    reg_5d   = reg_df[reg_df['Target_Horizon'] == '5d'].copy()
    colors4  = ['#AAAAAA' if 'Baseline' in m else '#E8714A'
                for m in reg_5d['Model']]
    bars4 = ax4.bar(range(len(reg_5d)), reg_5d['RMSE'],
                    color=colors4, alpha=0.85)
    r5_min = reg_5d['RMSE'].min()
    r5_max = reg_5d['RMSE'].max()
    mg5    = (r5_max - r5_min) * 3 if r5_max > r5_min else 0.001
    ax4.set_ylim(r5_min - mg5, r5_max + mg5 * 4)
    ax4.set_xticks(range(len(reg_5d)))
    ax4.set_xticklabels(reg_5d['Model'], rotation=35, ha='right', fontsize=7)
    _add_bar_labels(ax4, bars4, fmt='{:.5f}', padding=mg5 * 0.3)
    ax4.set_title('Regression RMSE (5-day)')
    ax4.set_ylabel('RMSE')
    ax4.grid(axis='y', alpha=0.4)

    # ── 5. Direction Accuracy (1d) – 0.5 기준선 강조 ──────
    ax5 = fig.add_subplot(3, 3, 5)
    colors5 = ['#AAAAAA' if 'Baseline' in m else '#3A9EDE'
               for m in reg_1d['Model']]
    bars5 = ax5.bar(range(len(reg_1d)), reg_1d['Direction_Accuracy'],
                    color=colors5, alpha=0.85)
    da_min = reg_1d['Direction_Accuracy'].min()
    da_max = reg_1d['Direction_Accuracy'].max()
    ax5.set_ylim(min(da_min * 0.97, 0.47), max(da_max * 1.03, 0.56))
    ax5.axhline(0.5, color='red', linestyle='--', linewidth=1.5, label='Random (0.50)')
    ax5.set_xticks(range(len(reg_1d)))
    ax5.set_xticklabels(reg_1d['Model'], rotation=35, ha='right', fontsize=7)
    _add_bar_labels(ax5, bars5, fmt='{:.4f}', padding=0.001)
    ax5.set_title('Direction Accuracy (1-day)')
    ax5.set_ylabel('Accuracy')
    ax5.legend(fontsize=8)
    ax5.grid(axis='y', alpha=0.4)

    # ── 6. Top 5 분류 CV F1 ───────────────────────────────
    ax6 = fig.add_subplot(3, 3, 6)
    t5c    = top5['classification_top5']
    labels = [f"n={r.n_estimators}, d={r.max_depth}\nleaf={r.min_samples_leaf}, cw={r.class_weight}"
              for r in t5c.itertuples()]
    # 1위 강조
    colors6 = ['#1F77B4'] + ['#7FB3D9'] * (len(t5c) - 1)
    bars6   = ax6.bar(range(len(t5c)), t5c['CV_F1'], color=colors6, alpha=0.9)
    ax6.set_xticks(range(len(t5c)))
    ax6.set_xticklabels(labels, fontsize=6.5)
    ax6.set_ylim(t5c['CV_F1'].min() * 0.97, t5c['CV_F1'].max() * 1.04)
    _add_bar_labels(ax6, bars6, fmt='{:.4f}', padding=0.001)
    ax6.set_title('Top 5 Classification (CV F1)')
    ax6.set_ylabel('CV F1')
    ax6.grid(axis='y', alpha=0.4)

    # ── 7. Top 5 분류 Test F1 ─────────────────────────────
    ax7 = fig.add_subplot(3, 3, 7)
    colors7 = ['#D62728'] + ['#E8998D'] * (len(t5c) - 1)
    bars7   = ax7.bar(range(len(t5c)), t5c['Test_F1'], color=colors7, alpha=0.9)
    ax7.set_xticks(range(len(t5c)))
    ax7.set_xticklabels(labels, fontsize=6.5)
    ax7.set_ylim(t5c['Test_F1'].min() * 0.97, t5c['Test_F1'].max() * 1.04)
    _add_bar_labels(ax7, bars7, fmt='{:.4f}', padding=0.001)
    ax7.set_title('Top 5 Classification (Test F1)')
    ax7.set_ylabel('Test F1')
    ax7.grid(axis='y', alpha=0.4)

    # ── 8. Top 5 회귀 RMSE (가로 막대 + 범위 강조) ────────
    ax8 = fig.add_subplot(3, 3, 8)
    t5r     = top5['regression_1d_top5']
    colors8 = ['#6A3FA5'] + ['#A68DD1'] * (len(t5r) - 1)
    bars8   = ax8.barh(range(len(t5r)), t5r['RMSE'],
                       color=colors8, alpha=0.9)
    ax8.set_yticks(range(len(t5r)))
    ax8.set_yticklabels(t5r['Model'], fontsize=8)
    ax8.invert_yaxis()
    # 차이가 보이도록 x축 범위 좁힘
    r_min = t5r['RMSE'].min()
    r_max = t5r['RMSE'].max()
    mg8   = (r_max - r_min) * 5 if r_max > r_min else 0.0001
    ax8.set_xlim(r_min - mg8, r_max + mg8 * 8)
    _add_hbar_labels(ax8, bars8, fmt='{:.6f}', padding=mg8 * 0.3)
    ax8.set_title('Top 5 Regression RMSE (1-day)')
    ax8.set_xlabel('RMSE')
    ax8.grid(axis='x', alpha=0.4)

    # ── 9. Top 5 회귀 Direction Accuracy ──────────────────
    ax9 = fig.add_subplot(3, 3, 9)
    colors9 = ['#C0392B'] + ['#E88D84'] * (len(t5r) - 1)
    bars9   = ax9.barh(range(len(t5r)), t5r['Direction_Accuracy'],
                       color=colors9, alpha=0.9)
    ax9.set_yticks(range(len(t5r)))
    ax9.set_yticklabels(t5r['Model'], fontsize=8)
    ax9.invert_yaxis()
    da_vals  = t5r['Direction_Accuracy']
    da_range = da_vals.max() - da_vals.min()
    mg9 = da_range * 5 if da_range > 0 else 0.005
    ax9.set_xlim(da_vals.min() - mg9, da_vals.max() + mg9 * 8)
    ax9.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='Random (0.50)')
    _add_hbar_labels(ax9, bars9, fmt='{:.4f}', padding=mg9 * 0.3)
    ax9.set_title('Top 5 Regression Dir.Acc (1-day)')
    ax9.set_xlabel('Direction Accuracy')
    ax9.legend(fontsize=8)
    ax9.grid(axis='x', alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[시각화] 저장 완료: {save_path}")
    return fig


# ============================================================
# [최상위 함수] run_data_science_pipeline
# ============================================================

def run_data_science_pipeline(data_path, k_fold=5, save_plot=True):
    """
    S&P 500 데이터에 대해 전처리 → 분류 → 회귀 → Top 5 탐색을
    순서대로 수행하는 단일 최상위 파이프라인 함수입니다.

    Parameters
    ----------
    data_path  : str   CSV 파일 경로
    k_fold     : int   k-Fold 교차 검증 수 (기본 5)
    save_plot  : bool  결과 Figure를 파일로 저장할지 여부

    Returns
    -------
    dict  키:
        'data'               : 전처리 결과 dict
        'classification'     : 분류 결과 dict
        'regression'         : pd.DataFrame  모든 회귀 결과
        'top5'               : dict  Top 5 결과
    """
    print("=" * 60)
    print("  S&P 500 Data Science Pipeline 시작")
    print("=" * 60)

    # Step 1 – 전처리
    data = preprocess_data(data_path)

    # Step 2 – 분류
    cls_result = run_classification(data, k=k_fold)

    # Step 3 – 회귀
    reg_df = run_regression(data)

    # Step 4 – Top 5
    top5 = find_top5_combinations(data, k=k_fold)

    # Step 5 – 시각화
    if save_plot:
        fig = visualize_results(
            cls_result, reg_df, top5,
            feature_cols=data['feature_cols'],
            save_path='pipeline_results.png',
        )

    # ── 최종 요약 출력 ─────────────────────────────────────
    print("=" * 60)
    print("  최종 결과 요약")
    print("=" * 60)

    print("\n▶ 분류 (Test Set)")
    for k, v in cls_result['test_scores'].items():
        print(f"   {k:10s}: {v:.4f}")

    print("\n▶ 회귀 Top 모델 (1-day, RMSE 기준)")
    best_reg = top5['regression_1d_top5'].iloc[0]
    print(f"   모델: {best_reg['Model']}")
    print(f"   RMSE: {best_reg['RMSE']:.6f}")
    print(f"   Direction Accuracy: {best_reg['Direction_Accuracy']:.4f}")

    print("\n▶ Top 5 분류 조합 (CV F1 기준)")
    print(top5['classification_top5'][
        ['n_estimators','max_depth','min_samples_leaf','class_weight','CV_F1','Test_F1']
    ].to_string(index=False))

    print("\n▶ Top 5 회귀 조합 (RMSE 기준, 1-day)")
    print(top5['regression_1d_top5'][
        ['Model','MAE','RMSE','R2','Direction_Accuracy']
    ].to_string(index=False))

    print("\n" + "=" * 60)
    print("  Pipeline 완료!")
    print("=" * 60)

    return {
        'data'          : data,
        'classification': cls_result,
        'regression'    : reg_df,
        'top5'          : top5,
    }


# ============================================================
# 결과 저장 / 로드 유틸리티
# ============================================================

def save_results(results, path='pipeline_results.pkl'):
    """
    전체 파이프라인 결과를 pickle 파일로 저장합니다.
    다음 실행 시 모델을 다시 학습하지 않고 시각화만 할 수 있습니다.

    Parameters
    ----------
    results : dict   run_data_science_pipeline() 반환값
    path    : str    저장 경로 (기본 'pipeline_results.pkl')
    """
    import pickle
    # data 키 안의 numpy 배열은 그대로 저장 가능
    save_obj = {
        'classification': results['classification'],
        'regression'    : results['regression'],
        'top5'          : results['top5'],
        'feature_cols'  : results['data']['feature_cols'],
    }
    with open(path, 'wb') as f:
        pickle.dump(save_obj, f)
    print(f"[저장] 결과 저장 완료: {path}")


def load_and_visualize(pkl_path='pipeline_results.pkl',
                       save_path='pipeline_results.png'):
    """
    저장된 결과를 불러와 시각화만 빠르게 실행합니다.
    모델 학습 없이 수 초 안에 완료됩니다.

    Parameters
    ----------
    pkl_path  : str   save_results()로 저장한 pkl 파일 경로
    save_path : str   저장할 이미지 파일 경로
    """
    import pickle
    print(f"[로드] {pkl_path} 불러오는 중 ...")
    with open(pkl_path, 'rb') as f:
        saved = pickle.load(f)

    visualize_results(
        cls_result  = saved['classification'],
        reg_df      = saved['regression'],
        top5        = saved['top5'],
        feature_cols= saved['feature_cols'],
        save_path   = save_path,
    )
    print("[완료] 시각화 재생성 완료!")


# ============================================================
# 실행 진입점
# ============================================================

if __name__ == '__main__':
    import sys

    # ── 시각화만 다시 실행할 때: python pipeline.py viz ───
    if len(sys.argv) > 1 and sys.argv[1] == 'viz':
        load_and_visualize(
            pkl_path  = 'pipeline_results.pkl',
            save_path = 'pipeline_results.png',
        )

    # ── 전체 파이프라인 실행 (기본) ───────────────────────
    else:
        results = run_data_science_pipeline(
            data_path='all_stocks_5yr.csv',
            k_fold=5,
            save_plot=True,
        )
        # 결과 저장 → 다음번 시각화만 실행 시 사용
        save_results(results, path='pipeline_results.pkl')