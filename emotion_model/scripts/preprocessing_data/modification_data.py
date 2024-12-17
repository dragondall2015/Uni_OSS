
### 데이터 확인 및 오버 샘플링

import pandas as pd

# CSV 파일 경로 설정
csv_path1 = "../../data_set/all_emotions_combined.csv"  # 원본 데이터 파일
output_csv_path = "../../data_set/all_emotions_oversampled.csv"  # 저장할 파일 경로

# CSV 파일 읽기
data = pd.read_csv(csv_path1)

# 열 이름 확인 (Unnamed 처리)
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])  # 불필요한 열 삭제

# 감정별 고유값 및 개수 확인
print("감정별 샘플 개수:")
print(data['emotion'].value_counts())

# 감정별 최대 샘플 수 확인
max_count = data['emotion'].value_counts().max()

# 오버 샘플링 수행 (각 감정을 최대 샘플 수에 맞게 복제)
balanced_data = data.groupby('emotion', group_keys=False).apply(lambda x: x.sample(max_count, replace=True)).reset_index(drop=True)

# 결과 확인
print("\n오버 샘플링 후 감정별 샘플 개수:")
print(balanced_data['emotion'].value_counts())

# 오버 샘플링된 데이터 저장
balanced_data.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
print(f"\n오버 샘플링된 데이터가 '{output_csv_path}'에 저장되었습니다.")
