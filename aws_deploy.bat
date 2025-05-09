@echo off
echo AWS 마이그레이션 배포 도구 시작...

REM 필요한 파일이 있는지 확인
if not exist fastapi_app_aws.py (
    echo fastapi_app_aws.py 파일이 없습니다.
    exit /b 1
)

REM 임시 배포 폴더 생성
mkdir aws_deploy
cd aws_deploy

REM AWS 배포용 파일 복사
echo 파일 복사 중...
copy ..\fastapi_app_aws.py application.py
copy ..\requirements.txt requirements.txt
copy ..\text_mining_analysis.py text_mining_analysis.py
copy ..\railway.json railway.json
xcopy ..\static static\ /E /I
xcopy ..\templates templates\ /E /I
xcopy ..\data data\ /E /I /Y

REM .ebextensions 폴더 생성
mkdir .ebextensions
echo "option_settings:" > .ebextensions/01_environment.config
echo "  aws:elasticbeanstalk:application:environment:" >> .ebextensions/01_environment.config
echo "    PYTHONPATH: /var/app/current" >> .ebextensions/01_environment.config
echo "    MAX_WORKERS: 4" >> .ebextensions/01_environment.config
echo "    MEMORY_LIMIT_MB: 2048" >> .ebextensions/01_environment.config

REM AWS EB CLI 실행
echo AWS Elastic Beanstalk 배포 준비...
echo 이제 다음 명령어를 실행하세요:
echo.
echo eb init -p python-3.9 textmining-app --region ap-northeast-2
echo eb create textmining-env
echo.

echo 준비 완료! aws_deploy 폴더에서 EB CLI 명령어를 실행하세요.
cd .. 