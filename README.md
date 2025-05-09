# 텍스트 마이닝 분석 애플리케이션

텍스트 데이터에 대한 고급 분석 기능을 제공하는 FastAPI 기반 웹 애플리케이션입니다.

## 주요 기능

- 텍스트 파일 분석 (CSV, TXT)
- 키워드 빈도 분석 및 워드클라우드
- TF-IDF 분석
- 토픽 모델링 (LDA)
- 감정 분석
- 네트워크 분석 및 시각화
- 3D 클러스터링 시각화

## 배포 환경

현재 두 가지 버전이 관리되고 있습니다:

1. **Railway/Render 배포용**: `fastapi_app.py`
   - 메모리 최적화를 위해 3D 시각화 기능 제거
   - 512MB RAM 제한 환경에 최적화

2. **AWS 배포용**: `fastapi_app_aws.py`
   - 모든 기능 포함 (3D 시각화 포함)
   - 최소 2GB RAM 권장 (t2.small)

## AWS 배포 방법

### 사전 준비
1. AWS CLI 설치 및 구성
2. 프로젝트 클론

### 배포 단계

1. AWS Elastic Beanstalk 콘솔 방문
   - [AWS 콘솔](https://console.aws.amazon.com/)에서 Elastic Beanstalk 서비스 선택

2. 새 애플리케이션 생성
   - 애플리케이션 이름: `textmining-app`
   - 플랫폼: Python
   - 플랫폼 버전: Python 3.9
   - 애플리케이션 코드: 코드 업로드

3. 배포 패키지 준비
   - `aws_deploy.bat` 스크립트를 실행하여 필요한 파일 복사
   - `aws_deploy` 폴더 내용을 ZIP 파일로 압축

4. 환경 구성
   - 인스턴스 유형: t2.small (2GB RAM) 이상 권장
   - 환경 변수 설정:
     - PYTHONPATH: /var/app/current
     - MAX_WORKERS: 4
     - MEMORY_LIMIT_MB: 2048

5. 환경 생성 완료 및 배포 확인

## 로컬 개발

```bash
# 가상환경 활성화
textmining_env\Scripts\activate

# 로컬 서버 실행
python fastapi_app.py   # Railway/Render 버전
python fastapi_app_aws.py   # AWS 버전 (3D 시각화 포함)
``` 