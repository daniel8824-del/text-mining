import pandas as pd
import json

# CSV 파일 읽기
df = pd.read_csv('knu_sentiment_lexicon.csv')

# 딕셔너리로 변환
sentiment_dict = {}
for _, row in df.iterrows():
    word = row['word']
    score = float(row['score'])
    sentiment_dict[word] = score

# JSON 파일로 저장
with open('knu_sentiment_lexicon.json', 'w', encoding='utf-8') as f:
    json.dump(sentiment_dict, f, ensure_ascii=False, indent=2)

print(f"변환 완료: {len(sentiment_dict)}개의 단어가 JSON 파일로 저장되었습니다.") 