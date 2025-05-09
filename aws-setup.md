# AWS 마이그레이션 가이드

## 1. 사전 준비사항

- AWS 계정 생성 및 설정
- [AWS CLI](https://aws.amazon.com/cli/) 설치
- [EB CLI](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install.html) 설치 
- Python 3.9 이상

## 2. 마이그레이션 준비

### AWS IAM 사용자 생성
1. AWS Management Console에 로그인
2. IAM 서비스로 이동
3. 새 사용자 생성 (AdministratorAccess 권한 또는 필요한 권한만 부여)
4. 액세스 키 ID 및 비밀 액세스 키 저장

### AWS CLI 설정
```bash
aws configure
```
- AWS 액세스 키 ID 입력
- AWS 시크릿 액세스 키 입력
- 기본 리전: ap-northeast-2 (서울)
- 기본 출력 형식: json

## 3. Elastic Beanstalk 배포

### 자동화 스크립트 사용
1. `aws_deploy.bat` 실행
```bash
aws_deploy.bat
```

2. 생성된 aws_deploy 디렉토리로 이동
```bash
cd aws_deploy
```

3. EB 애플리케이션 초기화
```bash
eb init -p python-3.9 textmining-app --region ap-northeast-2
```

4. EB 환경 생성 및 배포
```bash
eb create textmining-env
```

### 수동 배포 (스크립트 사용 안할 경우)
1. fastapi_app_aws.py를 application.py로 복사
2. .ebextensions 폴더와 환경 설정 파일 생성
3. eb init 및 eb create 명령어 실행

## 4. 환경 변수 설정

Elastic Beanstalk 콘솔에서 다음 환경 변수 설정:
- PYTHONPATH: /var/app/current
- MAX_WORKERS: 4
- MEMORY_LIMIT_MB: 2048 (t2.small 인스턴스 기준)

## 5. 인스턴스 타입 선택

- **t2.micro (1GB)**: 가벼운 사용에 적합, 비용 최소화 (월 $8-10)
- **t2.small (2GB)**: 권장, 3D 시각화 작업 가능 (월 $17-20)
- **t3.medium (4GB)**: 대규모 데이터셋, 고성능 필요 시 (월 $30-35)

## 6. 문제 해결

- 로그 확인: `eb logs`
- SSH 접속: `eb ssh`
- 애플리케이션 재시작: `eb restart`

## 7. 배포 후 확인

1. `eb open` 명령으로 웹 애플리케이션 열기
2. 3D 시각화 포함한 모든 기능 정상 작동 확인 