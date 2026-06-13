import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 한글 폰트 설정 (보고서 시각화용 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic' if os.name == 'nt' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# STEP 01. 데이터 준비
# ==========================================
raw_data = sns.load_dataset('titanic')
df = raw_data[['survived', 'age', 'fare', 'sibsp', 'parch', 'sex', 'embarked', 'class']].copy()

print("=== STEP 01. 데이터 구조 확인 ===")
print(f"데이터 Shape: {df.shape}")
print(df.head(), "\n")

# ==========================================
# STEP 02. 탐색적 데이터 분석 (EDA) 및 시각화 파일 저장
# ==========================================
print("=== STEP 02. EDA 수행 및 시각화 이미지 생성 ===")
print("1. 결측치 비율:\n", df.isnull().mean() * 100)
print("\n2. 타겟 변수(생존율) 분포:\n", df['survived'].value_counts(normalize=True))

# 필수 시각화 4종을 하나의 이미지 파일로 저장
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(data=df, x='fare', hue='survived', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('1. Histogram: Fare 분포')

sns.boxplot(data=df, x='class', y='age', hue='survived', ax=axes[0, 1])
axes[0, 1].set_title('2. Boxplot: 클래스/나이별 이상치')

# 상관관계 분석을 위한 임시 수치화 수집
corr_df = df[['survived', 'age', 'fare', 'sibsp', 'parch']].corr()
sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1, 0])
axes[1, 0].set_title('3. Heatmap: 변수 간 상관관계')

sns.countplot(data=df, x='sex', hue='survived', ax=axes[1, 1])
axes[1, 1].set_title('4. Countplot: 성별 생존자 수')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=300) # 이 파일이 과제 제출용 시각화 자료가 됩니다.
print("-> 'eda_plots.png' 시각화 파일 저장 완료.\n")

# ==========================================
# STEP 03-4. 파생 변수 생성 (필수 2개 항목)
# ==========================================
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = np.where(df['family_size'] == 1, 1, 0)

# 특성 레이어 정의
numeric_features = ['age', 'fare', 'sibsp', 'parch', 'family_size']
categorical_features = ['sex', 'embarked', 'class']

X = df[numeric_features + categorical_features + ['is_alone']]
y = df['survived']

# 데이터 분할 (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# STEP 03, 04, 05. 파이프라인 실험 구현
# ==========================================
results = []

def evaluate_pipeline(exp_name, num_imp, cat_imp, encoder, scaler, use_fs, model_name, model_obj):
    if exp_name == 'Base':
        # 지시서 조건: 없음/없음/없음/없음 -> 결측치와 문자열이 없는 완전 무결한 독립변수로만 구성
        base_cols = ['sibsp', 'parch', 'family_size', 'is_alone']
        X_tr, X_te = X_train[base_cols], X_test[base_cols]
        
        model_obj.fit(X_tr, y_train)
        preds = model_obj.predict(X_te)
        probs = model_obj.predict_proba(X_te)[:, 1] if hasattr(model_obj, "predict_proba") else preds
    else:
        # 가산점 요건: Pipeline 및 ColumnTransformer 활용 구조
        num_transformer = Pipeline([('imputer', num_imp), ('scaler', scaler)])
        cat_transformer = Pipeline([('imputer', cat_imp), ('encoder', encoder)])
        
        preprocessor = ColumnTransformer([
            ('num', num_transformer, numeric_features),
            ('cat', cat_transformer, categorical_features)
        ], remainder='passthrough')
        
        steps = [('preprocessor', preprocessor)]
        if use_fs:
            steps.append(('selector', SelectKBest(score_func=f_classif, k=5)))
        steps.append(('model', model_obj))
        
        pipeline = Pipeline(steps)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else preds

    # 평가지표 산출 (+3점 가점 항목 자동 계산)
    return {
        '실험': exp_name, '모델': model_name,
        'Accuracy': round(accuracy_score(y_test, preds), 4),
        'Precision': round(precision_score(y_test, preds, zero_division=0), 4),
        'Recall': round(recall_score(y_test, preds), 4),
        'F1-score': round(f1_score(y_test, preds), 4),
        'ROC-AUC': round(roc_auc_score(y_test, probs), 4)
    }

# 대상 알고리즘 (Classification 최소 2개 필수)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# 실험 조합 루프 수행 (5번 테이블 요구조건 100% 매칭)
for m_n, m_o in models.items():
    results.append(evaluate_pipeline('Base', None, None, None, None, False, m_n, m_o))

for m_n, m_o in models.items():
    results.append(evaluate_pipeline('Exp-1', SimpleImputer(strategy='mean'), SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'), StandardScaler(), False, m_n, m_o))

for m_n, m_o in models.items():
    results.append(evaluate_pipeline('Exp-2', SimpleImputer(strategy='median'), SimpleImputer(strategy='most_frequent'), OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), MinMaxScaler(), True, m_n, m_o))

for m_n, m_o in models.items():
    results.append(evaluate_pipeline('Exp-3', SimpleImputer(strategy='most_frequent'), SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'), RobustScaler(), True, m_n, m_o))

# 최종 결과 콘솔 출력용
summary_df = pd.DataFrame(results)
print("=== 5. 실험 비교 항목 결과 표 ===")
print(summary_df.to_string(index=False))