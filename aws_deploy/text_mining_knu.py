import os  # 파일 경로 조작
import pandas as pd  # 데이터 프레임 조작
import requests  # HTTP 요청
import zipfile  # zip 파일 압축 해제
import io  # 메모리 기반 파일 시스템

def download_knu_sentiment_dict(save_folder='.'):
    """KNU 한국어 감성사전을 다운로드하고 CSV 형식으로 변환합니다."""
    print("KNU 한국어 감성사전을 다운로드합니다...")
    
    # Github에서 제공하는 KNU 감성사전 다운로드
    url = "https://github.com/park1200656/KnuSentiLex/archive/refs/heads/master.zip"
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    
    # zip 파일 압축 해제
    temp_folder = os.path.join(save_folder, "temp_knu")
    os.makedirs(temp_folder, exist_ok=True)
    z.extractall(temp_folder)
    
    # 감성사전 파일 경로 및 로드
    sentiment_path = os.path.join(temp_folder, "KnuSentiLex-master", "data", "SentiWord_info.json")
    print(f"감성사전 파일: {sentiment_path}")
    
    # JSON 파일 로드 및 처리
    import json
    with open(sentiment_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 감성사전 데이터 추출
    words = []
    scores = []
    
    print("JSON 파일에서 감성 단어와 점수를 추출합니다...")
    for item in data:
        word = item.get('word')
        polarity = item.get('polarity')
        if word and polarity is not None:
            words.append(word)
            scores.append(float(polarity))
    
    # 데이터프레임 생성
    df = pd.DataFrame({"word": words, "score": scores})
    
    # CSV 파일로 저장
    csv_path = os.path.join(save_folder, "knu_sentiment_lexicon.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"KNU 감성사전이 {csv_path}에 저장되었습니다. (총 {len(df)}개 단어)")
    return True

class KnuTextMining:
    def __init__(self, sentiment_dict_path='knu_sentiment_lexicon.csv'):
        """KNU 감성사전을 로드합니다."""
        self.sentiment_dict = pd.read_csv(sentiment_dict_path, encoding='utf-8-sig')
        self.sentiment_dict = self.sentiment_dict.set_index('word')['score'].to_dict()
    
    def get_sentiment_score(self, word):
        """단어의 감성 점수를 반환합니다."""
        return self.sentiment_dict.get(word, 0)
    
    def analyze_text(self, text):
        """텍스트의 전체 감성 점수를 계산합니다."""
        words = text.split()
        scores = [self.get_sentiment_score(word) for word in words]
        return sum(scores) / len(words) if words else 0

if __name__ == "__main__":
    download_knu_sentiment_dict() 