# your_app/generate_dummy_data.py
import pandas as pd
import random

# 더미 데이터 생성
data = {
    'category': ['A', 'B', 'C', 'D'],
    'value': [random.randint(1, 10) for _ in range(4)]
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# CSV 파일로 저장
df.to_csv('dummy_data.csv', index=False)
