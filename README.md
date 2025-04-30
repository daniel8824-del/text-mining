# 한국어 텍스트 마이닝 분석 도구

이 프로젝트는 한국어 텍스트 데이터의 다양한 분석을 수행하는 웹 애플리케이션입니다.

## 주요 기능

- 키워드 추출 및 빈도 분석
- TF-IDF 분석
- 토픽 모델링 (LDA)
- 감정 분석
- 워드클라우드 생성
- 키워드 네트워크 분석
- 키워드 상관관계 분석
- 중심성 분석
- 클러스터링 분석

## 설치 방법

```bash
# 저장소 클론
git clone https://github.com/jk0601/LDA_TEXT.git
cd LDA_TEXT

# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 앱 실행
python app.py
```

## 사용 방법

1. CSV 파일을 업로드합니다.
2. 텍스트 데이터가 포함된 컬럼명을 입력합니다.
3. 품사 태깅 옵션을 선택합니다.
4. 불용어를 설정합니다 (선택사항).
5. '분석 시작' 버튼을 클릭합니다.
6. 분석 결과를 확인하고 다운로드할 수 있습니다.

## 배포

이 애플리케이션은 streamlit을 사용하여 배포할 수 있습니다.

## 라이센스

MIT 라이센스

## 연락처

- 문의사항: jkj0601@gmail.com 
