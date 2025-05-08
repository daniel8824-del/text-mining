FROM python:3.9

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    g++ \
    default-jdk \
    python3-dev \
    fonts-nanum \
    curl

# 한국어 환경 설정
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Java 환경 변수 설정
ENV JAVA_HOME=/usr/lib/jvm/default-java

# 작업 디렉토리 설정
WORKDIR /app

# 요구사항 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK 데이터 다운로드
RUN python -m nltk.downloader punkt stopwords wordnet

# 폰트 디렉토리 생성
RUN mkdir -p /app/fonts
# 시스템 폰트를 애플리케이션 폴더로 복사
RUN cp /usr/share/fonts/truetype/nanum/NanumGothic.ttf /app/fonts/ || echo "Font not found, will use default font"

# 애플리케이션 코드 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p /app/uploads /app/static

# 파일 권한 설정
RUN chmod -R 777 /app/uploads /app/static

# 포트 설정
EXPOSE 8000

# 실행 명령
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"] 