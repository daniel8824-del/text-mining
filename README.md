# 텍스트 마이닝 분석 도구

이 프로젝트는 한국어 텍스트 데이터를 분석하기 위한 웹 기반 텍스트 마이닝 도구입니다. LDA 토픽 모델링, 감성 분석, 단어 빈도 분석 등 다양한 텍스트 분석 기능을 제공합니다.

## 주요 기능

- **토픽 모델링**: LDA(Latent Dirichlet Allocation) 알고리즘을 활용한 토픽 추출
- **감성 분석**: KNU 감성 사전 기반 텍스트 감성 분석
- **단어 빈도 분석**: 워드클라우드, 빈도 차트 등 시각화 도구
- **텍스트 전처리**: 불용어 제거, 형태소 분석 등 한국어 특화 전처리

## 기술 스택

- **백엔드**: Python, FastAPI
- **데이터 분석**: scikit-learn, gensim, matplotlib, pandas
- **자연어 처리**: konlpy, NLTK
- **프론트엔드**: HTML, CSS, JavaScript, Bootstrap
- **배포**: Docker, AWS, Render, Railway

## 설치 및 실행

### 로컬 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv textmining_env
source textmining_env/bin/activate  # Windows: textmining_env\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 애플리케이션 실행
python fastapi_app.py
```

### Docker를 이용한 실행

```bash
docker-compose up
```

## 클라우드 배포 옵션

이 프로젝트는 다양한 클라우드 서비스에 배포할 수 있도록 설정되어 있습니다.

### Amazon AWS

- AWS Elastic Beanstalk 또는 EC2를 통한 배포 가능
- 유료 서비스로, 사용량에 따른 과금 발생
- 안정적인 서비스와 확장성 제공

### Render

- `render.yaml` 파일을 통한 쉬운 배포
- 무료 티어 존재하나 대용량 데이터 처리 시 높은 과금 발생 가능
- CI/CD 파이프라인 지원

### Railway

- `railway.json` 파일을 통한 간편 배포
- 초기 무료 크레딧 제공, 이후 사용량에 따른 과금
- 대용량 분석 작업 수행 시 높은 과금 발생 가능

## 주의사항

- 대용량 데이터 처리 시 클라우드 서비스(Render, Railway, AWS 등)에서 높은 과금이 발생할 수 있습니다.
- 배포 전 각 서비스의 가격 정책을 확인하고 예산에 맞는 옵션을 선택하세요.
- 대용량 분석 작업은 가능한 로컬 환경에서 수행하는 것을 권장합니다.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요. 