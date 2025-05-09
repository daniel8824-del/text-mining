import sys
import os
import asyncio
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request, BackgroundTasks

import pandas as pd
import shutil
from typing import List, Optional
import uvicorn
from text_mining_analysis import TextMiningAnalysis
import matplotlib.pyplot as plt
import zipfile
import io
import tempfile
import uuid
import json
import traceback
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import TSNE
import mpl_toolkits.mplot3d as plt3d
from matplotlib.colors import ListedColormap
import asyncio
import psutil
import gc
import threading
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fastapi_app")
logger.info("애플리케이션 시작 중...")

# 한글 폰트 설정 (matplotlib)
import matplotlib
matplotlib.use('Agg')  # 백엔드 설정
try:
    import matplotlib.font_manager as fm
    # 가능한 한글 폰트 경로 목록
    font_paths = [
        'C:/Windows/Fonts/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/app/fonts/NanumGothic.ttf'
    ]
    
    font_installed = False
    for font_path in font_paths:
        if os.path.exists(font_path):
            # Malgun Gothic 대신 항상 NanumGothic으로 설정
            plt.rcParams['font.family'] = 'NanumGothic'
            plt.rcParams['axes.unicode_minus'] = False
            logger.info(f"한글 폰트 설정 완료: {font_path}")
            font_installed = True
            break
            
    if not font_installed:
        logger.warning("사용 가능한 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
except Exception as e:
    logger.error(f"폰트 설정 중 오류 발생: {e}")

# 앱 생성
app = FastAPI(title="텍스트 마이닝 분석 API", description="한국어 텍스트 마이닝 분석을 위한 API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (배포 시 제한 필요)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 현재 디렉토리 경로를 기준으로 필요한 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

# 디렉토리 생성
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# 디렉토리 권한 설정 (Linux/Unix 환경에서만 작동)
try:
    import stat
    os.chmod(STATIC_DIR, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 777 권한
    os.chmod(UPLOAD_DIR, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 777 권한
    logger.info("디렉토리 권한 설정 완료")
except Exception as e:
    logger.error(f"디렉토리 권한 설정 중 오류: {e}")

# 템플릿 및 정적 파일 설정
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 품사 태깅 옵션
POS_OPTIONS = {
    'Noun': '명사',
    'Verb': '동사',
    'Adjective': '형용사',
    'Adverb': '부사',
    'Determiner': '관형사'
}

# 메모리 관리 설정
IS_RAILWAY = 'RAILWAY_ENVIRONMENT' in os.environ or 'RAILWAY_SERVICE_NAME' in os.environ
MEMORY_LIMIT_MB = int(os.environ.get('MEMORY_LIMIT_MB', '500'))  # 기본값 500MB, 환경 변수에서 가져옴
MAX_MEMORY_PERCENT = 80  # 최대 메모리 사용률 (80%)
MEMORY_THRESHOLD = MEMORY_LIMIT_MB * 0.8 * 1024 * 1024  # 메모리 임계값 (지정된 한도의 80%)
MEMORY_CHECK_INTERVAL = 20  # 20초마다 메모리 체크
memory_monitor_running = False

logger.info(f"메모리 제한: {MEMORY_LIMIT_MB}MB, 임계값: {MEMORY_THRESHOLD/(1024*1024):.1f}MB")
logger.info(f"레일웨이 환경 감지: {IS_RAILWAY}")

# 부하를 줄이기 위한 수단 - 무거운 모듈 지연 로딩
def load_heavy_modules():
    """무거운 모듈을 필요할 때만 로드하여 초기 메모리 사용량 감소"""
    logger.info("무거운 모듈 로딩 시작...")
    
    start_time = time.time()
    start_memory = print_memory_info()
    
    # 이 함수에서만 사용되는 모듈 임포트
    global matplotlib, plt, np, nx, KMeans, SpectralClustering, TSNE
    global plt3d, ListedColormap
    
    # matplotlib 관련
    if 'matplotlib' not in sys.modules:
        import matplotlib
        matplotlib.use('Agg')  # 백엔드 설정
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d as plt3d
        from matplotlib.colors import ListedColormap
        
        # 한글 폰트 설정 (matplotlib)
        try:
            import matplotlib.font_manager as fm
            # 가능한 한글 폰트 경로 목록
            font_paths = [
                'C:/Windows/Fonts/NanumGothic.ttf',
                '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                '/app/fonts/NanumGothic.ttf'
            ]
            
            font_installed = False
            for font_path in font_paths:
                if os.path.exists(font_path):
                    # Malgun Gothic 대신 항상 NanumGothic으로 설정
                    plt.rcParams['font.family'] = 'NanumGothic'
                    plt.rcParams['axes.unicode_minus'] = False
                    logger.info(f"한글 폰트 설정 완료: {font_path}")
                    font_installed = True
                    break
                    
            if not font_installed:
                logger.warning("사용 가능한 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        except Exception as e:
            logger.error(f"폰트 설정 중 오류 발생: {e}")
    
    # 데이터 분석 관련 라이브러리
    if 'numpy' not in sys.modules:
        import numpy as np
    
    if 'networkx' not in sys.modules:
        import networkx as nx
        
    # 머신러닝 관련
    if 'sklearn.cluster' not in sys.modules:
        from sklearn.cluster import KMeans, SpectralClustering
        
    if 'sklearn.manifold' not in sys.modules:
        from sklearn.manifold import TSNE
    
    # 메모리 사용량 체크
    end_time = time.time()
    end_memory = print_memory_info()
    
    logger.info(f"무거운 모듈 로딩 완료: {end_time - start_time:.2f}초 소요")
    logger.info(f"모듈 로딩으로 인한 메모리 증가: {end_memory - start_memory:.2f}MB")
    
    return True

# 모듈 로딩 상태
heavy_modules_loaded = False

# 메모리 사용률 계산 함수
def get_memory_usage_percent():
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # 시스템 메모리 정보 가져오기
    system_memory = psutil.virtual_memory()
    
    # 프로세스 메모리 사용률 계산
    process_percent = (memory_info.rss / system_memory.total) * 100
    
    return process_percent

# 메모리 모니터링 함수
async def monitor_memory():
    global memory_monitor_running
    
    if memory_monitor_running:
        return
    
    memory_monitor_running = True
    logger.info("메모리 모니터링 시작됨")
    
    try:
        while True:
            process = psutil.Process()
            memory_info = process.memory_info()
            current_memory = memory_info.rss
            memory_percent = get_memory_usage_percent()
            
            # 메모리 사용률이 임계값을 초과하면 정리 수행
            if memory_percent > MAX_MEMORY_PERCENT or current_memory > MEMORY_THRESHOLD:
                logger.warning(f"경고! 메모리 사용량 ({current_memory / 1024 / 1024:.2f} MB, {memory_percent:.1f}%)이 임계값을 초과하여 정리합니다.")
                clean_memory()
                
                # 메모리 정리 후에도 여전히 높은 경우 추가 조치
                if get_memory_usage_percent() > MAX_MEMORY_PERCENT:
                    logger.warning("메모리 사용률이 여전히 높습니다. 추가 정리 작업 수행...")
                    # 추가 메모리 정리: 이미지 캐시 정리
                    plt.close('all')  # 열려있는 모든 matplotlib 그림 닫기
                    
                    # 파이썬 자체 캐시 정리
                    for module in list(sys.modules.values()):
                        if hasattr(module, 'clear_cache'):
                            try:
                                module.clear_cache()
                            except:
                                pass
                    
                    # 임시 파일 정리 (최근 생성된 파일도 포함)
                    try:
                        for root, dirs, files in os.walk(STATIC_DIR):
                            for file in files:
                                if (file.startswith("wordcloud_") or 
                                    file.startswith("network_") or 
                                    file.startswith("temp_") or 
                                    file.startswith("bubble_") or 
                                    file.startswith("clustering_")):
                                    try:
                                        file_path = os.path.join(root, file)
                                        # 1시간 이상 된 파일만 삭제
                                        if (time.time() - os.path.getctime(file_path)) > 3600:
                                            os.remove(file_path)
                                            logger.info(f"오래된 임시 파일 삭제: {file}")
                                    except Exception as e:
                                        logger.error(f"파일 삭제 오류: {e}")
                    except Exception as e:
                        logger.error(f"임시 파일 정리 오류: {e}")
                    
                    # 마지막 수단: 강제 가비지 컬렉션
                    gc.collect(generation=2)
            
            # 주기적으로 메모리 상태 로깅
            logger.info(f"현재 메모리 사용량: {current_memory / 1024 / 1024:.2f} MB ({memory_percent:.1f}%)")
            
            # 일정 시간 대기
            await asyncio.sleep(MEMORY_CHECK_INTERVAL)
    except Exception as e:
        logger.error(f"메모리 모니터링 오류: {e}")
        logger.error(traceback.format_exc())
    finally:
        memory_monitor_running = False
        # 오류 발생 시 재시작
        await asyncio.sleep(5)
        asyncio.create_task(monitor_memory())

# 앱 시작 시 메모리 모니터링 시작
@app.on_event("startup")
async def startup_event():
    # 백그라운드 작업으로 메모리 모니터링 시작
    asyncio.create_task(monitor_memory())
    logger.info("애플리케이션이 시작되었습니다.")

# 애플리케이션 초기화 시 메모리 정보 출력
def print_memory_info():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logger.info(f"메모리 사용량: {memory_mb:.2f} MB / {MEMORY_LIMIT_MB} MB (유료 사용자)")
    return memory_mb

print_memory_info()

# 메모리 정리 함수
def clean_memory():
    # 메모리 정리 전 사용량 출력
    logger.info("메모리 정리 시작")
    process = psutil.Process()
    before_memory = process.memory_info().rss / 1024 / 1024
    logger.info(f"정리 전 메모리 사용량: {before_memory:.2f} MB")
    
    # 메모리 정리
    gc.collect()
    
    # 메모리 정리 후 사용량 출력
    after_memory = process.memory_info().rss / 1024 / 1024
    logger.info(f"정리 후 메모리 사용량: {after_memory:.2f} MB")
    logger.info(f"정리된 메모리: {before_memory - after_memory:.2f} MB")

# 레일웨이 환경에서의 안전한 최대 파일 크기 설정
if IS_RAILWAY:
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 레일웨이에서는 20MB로 제한
    MAX_CHART_SIZE = 30  # 레일웨이에서는 항목 수 축소
    logger.info("레일웨이 환경이 감지되어 메모리 사용량을 제한합니다.")
else:
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 다른 환경에서는 100MB로 유지
    MAX_CHART_SIZE = 50  # 다른 환경에서는 항목 수 유지

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "pos_options": POS_OPTIONS})

@app.post("/analyze")
async def analyze_text(
    file: UploadFile = File(...),
    text_column: str = Form(...),
    pos_tags: Optional[List[str]] = Form(None),
    stopwords: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    # 현재 메모리 상태 확인
    memory_percent = get_memory_usage_percent()
    if memory_percent > MAX_MEMORY_PERCENT:
        # 메모리 사용률이 너무 높으면 요청 거부
        clean_memory()  # 먼저 메모리 정리 시도
        
        # 정리 후에도 여전히 높으면 요청 거부
        if get_memory_usage_percent() > MAX_MEMORY_PERCENT:
            raise HTTPException(
                status_code=503, 
                detail="서버 부하가 높습니다. 잠시 후 다시 시도해주세요."
            )
    
    # 파일 크기 제한 확인
    file_contents = await file.read()
    if len(file_contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"파일 크기가 너무 큽니다. {MAX_FILE_SIZE // (1024 * 1024)}MB 이하의 파일을 업로드하세요.")
    
    # 파일 포인터 위치 리셋
    file.file.seek(0)
    
    # 파일 유효성 검사
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV 파일만 지원합니다.")

    file_path = None
    temp_files = []  # 정리해야 할 임시 파일 목록
    
    try:
        # 고유한 파일명 생성 (UUID 사용)
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # 파일 저장
        with open(file_path, "wb") as buffer:
            buffer.write(file_contents)
        
        # 메모리에서 파일 내용 제거
        del file_contents
        gc.collect()
        
        # 사용자 정의 불용어 처리
        custom_stopwords = []
        if stopwords:
            custom_stopwords = [word.strip() for word in stopwords.split(',') if word.strip()]
        
        # 텍스트 마이닝 분석 실행
        analyzer = TextMiningAnalysis(file_path=file_path, text_column=text_column)
        
        # 품사 태깅 및 전처리
        pos_filter = pos_tags if pos_tags else None
        analyzer.preprocess_text(pos_filter=pos_filter, custom_stopwords=custom_stopwords)
        
        # 메모리 정보 출력
        print_memory_info()
        
        # 분석 결과 얻기
        results = {}
        
        # 각 분석 단계마다 메모리 정리
        def checkpoint_cleanup():
            gc.collect()
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(f"체크포인트 메모리 사용량: {memory_info.rss / 1024 / 1024:.2f} MB")
        
        # 1. 키워드 추출
        try:
            keywords = analyzer.extract_keywords(top_n=20)
            results['keywords'] = [{'word': word, 'freq': freq} for word, freq in keywords.items()]
            checkpoint_cleanup()
        except Exception as e:
            logger.error(f"키워드 추출 오류: {e}")
            logger.error(traceback.format_exc())
            results['keywords'] = []
        
        # 2. TF-IDF 분석
        try:
            analyzer.perform_tf_idf_analysis()
            tfidf_keywords = analyzer.get_top_tf_idf_keywords(top_n=20)
            
            # TF-IDF 결과 통합
            all_tfidf_keywords = {}
            for doc_keywords in tfidf_keywords:
                for word, score in doc_keywords:
                    if word in all_tfidf_keywords:
                        all_tfidf_keywords[word] = max(all_tfidf_keywords[word], score)
                    else:
                        all_tfidf_keywords[word] = score
            
            results['tfidf_keywords'] = [
                {'word': word, 'score': float(score)} 
                for word, score in sorted(all_tfidf_keywords.items(), key=lambda x: x[1], reverse=True)[:20]
            ]
            checkpoint_cleanup()
        except Exception as e:
            logger.error(f"TF-IDF 분석 오류: {e}")
            logger.error(traceback.format_exc())
            results['tfidf_keywords'] = []
        
        # 3. 토픽 모델링
        try:
            topics = analyzer.topic_modeling(num_topics=5, num_words=10)
            results['topics'] = []
            for i, topic in enumerate(topics):
                topic_words = [{'word': word, 'score': float(score)} for word, score in topic]
                results['topics'].append({
                    'topic_id': i + 1,
                    'words': topic_words
                })
            checkpoint_cleanup()
        except Exception as e:
            logger.error(f"토픽 모델링 오류: {e}")
            logger.error(traceback.format_exc())
            results['topics'] = []
        
        # 4. 감정 분석
        try:
            # 감정 분석 전에 knu_dict가 제대로 로드되었는지 확인
            if not hasattr(analyzer, 'knu_dict') or not analyzer.knu_dict:
                logger.warning("감정 사전이 로드되지 않았습니다. 기본값을 사용합니다.")
                results['sentiment'] = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
            else:
                sentiment = analyzer.sentiment_analysis()
                if isinstance(sentiment, pd.DataFrame) and not sentiment.empty:
                    sentiment_counts = sentiment['Sentiment'].value_counts()
                    results['sentiment'] = {
                        'positive': int(sentiment_counts.get('Positive', 0)),
                        'negative': int(sentiment_counts.get('Negative', 0)),
                        'neutral': int(sentiment_counts.get('Neutral', 0))
                    }
                else:
                    logger.warning("빈 감정 분석 결과")
                    results['sentiment'] = {
                        'positive': 0,
                        'negative': 0,
                        'neutral': 0
                    }
            checkpoint_cleanup()
        except Exception as sentiment_error:
            logger.error(f"감정 분석 오류: {sentiment_error}")
            logger.error(traceback.format_exc())
            results['sentiment'] = {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
        
        # 5. 워드클라우드 생성
        try:
            # 폰트 경로 설정 (운영체제별 처리)
            font_path = None
            if os.path.exists('C:/Windows/Fonts/NanumGothic.ttf'):  # Windows
                font_path = 'C:/Windows/Fonts/NanumGothic.ttf'
            elif os.path.exists('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'):  # Ubuntu with Nanum
                font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
            elif os.path.exists('/app/fonts/NanumGothic.ttf'):  # Docker 환경
                font_path = '/app/fonts/NanumGothic.ttf'
            else:
                logger.warning("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
            
            # 시각화 결과 경로
            wordcloud_path = os.path.join(STATIC_DIR, f'wordcloud_{unique_filename}.png')
            logger.info(f"워드클라우드 저장 경로: {wordcloud_path}")
            
            try:
                # 워드클라우드 생성
                wordcloud = analyzer.create_word_cloud(font_path=font_path)
                
                # 워드클라우드 시각화 (제목 추가)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title('전체 키워드 워드클라우드', fontsize=16)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(wordcloud_path, bbox_inches='tight')
                plt.close()
                
                # 저장된 파일 확인
                if os.path.exists(wordcloud_path):
                    logger.info(f"워드클라우드 파일이 생성되었습니다: {wordcloud_path}")
                    results['wordcloud_path'] = f'/static/wordcloud_{unique_filename}.png'
                else:
                    logger.warning(f"워드클라우드 파일이 생성되지 않았습니다.")
                    results['wordcloud_path'] = ''
            except Exception as inner_error:
                logger.error(f"워드클라우드 이미지 저장 오류: {inner_error}")
                logger.error(traceback.format_exc())
                results['wordcloud_path'] = ''
        except Exception as wordcloud_error:
            logger.error(f"워드클라우드 생성 오류: {wordcloud_error}")
            logger.error(traceback.format_exc())
            results['wordcloud_path'] = ''
            
        # 6. 키워드 네트워크 분석
        try:
            network = analyzer.keyword_network_analysis(threshold=2, top_n=30)
            if network:
                # 네트워크 시각화 결과 경로
                network_path = os.path.join(STATIC_DIR, f'network_{unique_filename}.png')
                logger.info(f"네트워크 그래프 저장 경로: {network_path}")
                
                try:
                    # 한글 폰트 설정 - 명시적으로 여기서 다시 설정
                    try:
                        import matplotlib.font_manager as fm
                        # 가능한 한글 폰트 경로 목록
                        font_paths = [
                            'C:/Windows/Fonts/NanumGothic.ttf',
                            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                            '/app/fonts/NanumGothic.ttf'
                        ]
                        
                        font_set = False
                        for font_path in font_paths:
                            if os.path.exists(font_path):
                                plt.rcParams['font.family'] = 'NanumGothic'
                                plt.rcParams['axes.unicode_minus'] = False
                                logger.info(f"네트워크 그래프용 한글 폰트 설정 완료: {font_path}")
                                font_set = True
                                break
                                
                        if not font_set:
                            logger.warning("네트워크 그래프용 한글 폰트를 찾을 수 없습니다.")
                    except Exception as font_err:
                        logger.error(f"네트워크 그래프 폰트 설정 오류: {font_err}")
                    
                    # 네트워크 시각화
                    plt.figure(figsize=(10, 8))
                    analyzer.plot_network(network)
                    
                    # 네트워크 그래프 저장
                    plt.savefig(network_path, bbox_inches='tight')
                    plt.close()
                    
                    # 저장된 파일 확인
                    if os.path.exists(network_path):
                        logger.info(f"네트워크 그래프 파일이 생성되었습니다: {network_path}")
                        results['network_path'] = f'/static/network_{unique_filename}.png'
                    else:
                        logger.warning(f"네트워크 그래프 파일이 생성되지 않았습니다.")
                        results['network_path'] = ''
                except Exception as inner_error:
                    logger.error(f"네트워크 그래프 이미지 저장 오류: {inner_error}")
                    logger.error(traceback.format_exc())
                    results['network_path'] = ''
                
                # 네트워크 노드 정보 추가
                node_data = []
                for node in network.nodes():
                    node_data.append({
                        'word': node,
                        'size': network.nodes[node]['size']
                    })
                results['network_nodes'] = sorted(node_data, key=lambda x: x['size'], reverse=True)[:20]
            else:
                results['network_path'] = ''
                results['network_nodes'] = []
            checkpoint_cleanup()
        except Exception as network_error:
            logger.error(f"키워드 네트워크 분석 오류: {network_error}")
            logger.error(traceback.format_exc())
            results['network_path'] = ''
            results['network_nodes'] = []
        
        # 7. 추가 시각화 - 토픽 점유율 파이 차트
        try:
            # 토픽 분포 계산
            topic_distribution = analyzer.lda_model.transform(analyzer.tf_idf_matrix)
            topic_shares = topic_distribution.mean(axis=0)
            
            # 토픽 이름 생성
            topic_titles = []
            for topic in topics:
                total_weight = sum([weight for _, weight in topic[:3]])
                title_parts = []
                for word, weight in topic[:3]:
                    percentage = int(100 * weight / total_weight)
                    title_parts.append(f"{word}({percentage}%)")
                topic_titles.append(' + '.join(title_parts))
            
            topic_names = [f"토픽 {i+1}: {title}" for i, title in enumerate(topic_titles)]
            
            # 토픽 점유율 차트 저장 경로
            topic_chart_path = os.path.join(STATIC_DIR, f'topic_chart_{unique_filename}.png')
            
            # 한글 폰트 설정 - 명시적으로 여기서 다시 설정
            try:
                import matplotlib.font_manager as fm
                # 가능한 한글 폰트 경로 목록
                font_paths = [
                    'C:/Windows/Fonts/NanumGothic.ttf',
                    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                    '/app/fonts/NanumGothic.ttf'
                ]
                
                font_set = False
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        plt.rcParams['font.family'] = 'NanumGothic'
                        plt.rcParams['axes.unicode_minus'] = False
                        logger.info(f"파이 차트용 한글 폰트 설정 완료: {font_path}")
                        font_set = True
                        break
                        
                if not font_set:
                    logger.warning("파이 차트용 한글 폰트를 찾을 수 없습니다.")
            except Exception as font_err:
                logger.error(f"파이 차트 폰트 설정 오류: {font_err}")
            
            # 파이 차트 생성
            plt.figure(figsize=(10, 8))
            plt.pie(topic_shares, labels=topic_names, autopct='%1.1f%%', 
                   startangle=90, shadow=True)
            plt.title('토픽 점유율', fontsize=16)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(topic_chart_path, bbox_inches='tight')
            plt.close()
            
            if os.path.exists(topic_chart_path):
                # 변수명을 HTML 템플릿과 일치시킴
                results['topic_shares_path'] = f'/static/topic_chart_{unique_filename}.png'
            else:
                results['topic_shares_path'] = ''
        except Exception as e:
            logger.error(f"토픽 점유율 차트 생성 오류: {e}")
            logger.error(traceback.format_exc())
            results['topic_shares_path'] = ''
        
        # 8. 감정 분석 워드클라우드
        try:
            if hasattr(analyzer, 'pos_words') and hasattr(analyzer, 'neg_words'):
                # 긍정 단어 워드클라우드
                pos_word_counts = {}
                neg_word_counts = {}
                
                for doc in analyzer.corpus:
                    tokens = analyzer.tokenizer(doc)
                    for token in tokens:
                        if token in analyzer.pos_words:
                            pos_word_counts[token] = pos_word_counts.get(token, 0) + 1
                        elif token in analyzer.neg_words:
                            neg_word_counts[token] = neg_word_counts.get(token, 0) + 1
                
                from wordcloud import WordCloud
                
                # 긍정 워드클라우드 생성
                pos_cloud_path = os.path.join(STATIC_DIR, f'pos_cloud_{unique_filename}.png')
                pos_words_text = ' '.join([f"{word} " * count for word, count in pos_word_counts.items()])
                
                if pos_words_text.strip():
                    pos_cloud = WordCloud(
                        font_path=font_path,
                        width=800, height=400, 
                        background_color='white',
                        max_words=100,
                        colormap='YlGn'
                    ).generate(pos_words_text)
                    
                    plt.figure(figsize=(10, 5))
                    plt.imshow(pos_cloud, interpolation='bilinear')
                    plt.title('긍정 단어 워드클라우드')
                    plt.axis('off')
                    plt.savefig(pos_cloud_path, bbox_inches='tight')
                    plt.close()
                    
                    if os.path.exists(pos_cloud_path):
                        # 변수명을 HTML 템플릿과 일치시킴
                        results['pos_wordcloud_path'] = f'/static/pos_cloud_{unique_filename}.png'
                    else:
                        results['pos_wordcloud_path'] = ''
                else:
                    results['pos_wordcloud_path'] = ''
                
                # 부정 워드클라우드 생성
                neg_cloud_path = os.path.join(STATIC_DIR, f'neg_cloud_{unique_filename}.png')
                neg_words_text = ' '.join([f"{word} " * count for word, count in neg_word_counts.items()])
                
                if neg_words_text.strip():
                    neg_cloud = WordCloud(
                        font_path=font_path,
                        width=800, height=400, 
                        background_color='white',
                        max_words=100,
                        colormap='OrRd'
                    ).generate(neg_words_text)
                    
                    plt.figure(figsize=(10, 5))
                    plt.imshow(neg_cloud, interpolation='bilinear')
                    plt.title('부정 단어 워드클라우드')
                    plt.axis('off')
                    plt.savefig(neg_cloud_path, bbox_inches='tight')
                    plt.close()
                    
                    if os.path.exists(neg_cloud_path):
                        # 변수명을 HTML 템플릿과 일치시킴
                        results['neg_wordcloud_path'] = f'/static/neg_cloud_{unique_filename}.png'
                    else:
                        results['neg_wordcloud_path'] = ''
                else:
                    results['neg_wordcloud_path'] = ''
            else:
                results['pos_wordcloud_path'] = ''
                results['neg_wordcloud_path'] = ''
        except Exception as e:
            logger.error(f"감정 분석 워드클라우드 생성 오류: {e}")
            logger.error(traceback.format_exc())
            results['pos_wordcloud_path'] = ''
            results['neg_wordcloud_path'] = ''
        
        # 9. 히트맵 생성
        try:
            # 상위 15개 키워드 선택
            top_keywords = list(keywords.keys())[:15]
            
            # 키워드 상관관계 계산 (자카드 유사도)
            correlation_matrix = np.zeros((len(top_keywords), len(top_keywords)))
            
            for i, word1 in enumerate(top_keywords):
                for j, word2 in enumerate(top_keywords):
                    # 동시 출현 횟수 계산
                    co_occurrence = 0
                    for doc in analyzer.tokenized_corpus:
                        if word1 in doc and word2 in doc:
                            co_occurrence += 1
                    
                    # 자카드 유사도 계산
                    word1_docs = sum(1 for doc in analyzer.tokenized_corpus if word1 in doc)
                    word2_docs = sum(1 for doc in analyzer.tokenized_corpus if word2 in doc)
                    
                    if word1_docs + word2_docs - co_occurrence > 0:
                        jaccard = co_occurrence / (word1_docs + word2_docs - co_occurrence)
                    else:
                        jaccard = 0
                    
                    correlation_matrix[i, j] = jaccard
            
            # 히트맵 생성
            heatmap_path = os.path.join(STATIC_DIR, f'heatmap_{unique_filename}.png')
            plt.figure(figsize=(12, 10))
            
            # 한글 폰트 설정 확인
            try:
                import matplotlib.font_manager as fm
                plt.rcParams['font.family'] = 'NanumGothic'
                plt.rcParams['axes.unicode_minus'] = False
                logger.info(f"히트맵용 한글 폰트 설정 완료")
            except Exception as font_error:
                logger.error(f"폰트 설정 오류: {font_error}")
            
            # 히트맵 그리기
            import seaborn as sns
            sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', 
                       xticklabels=top_keywords, yticklabels=top_keywords)
            plt.title('키워드 상관관계 히트맵', fontsize=16)
            plt.tight_layout()
            plt.savefig(heatmap_path, bbox_inches='tight')
            plt.close()
            
            if os.path.exists(heatmap_path):
                results['heatmap_path'] = f'/static/heatmap_{unique_filename}.png'
            else:
                results['heatmap_path'] = ''
                
        except Exception as heatmap_error:
            logger.error(f"히트맵 생성 오류: {heatmap_error}")
            logger.error(traceback.format_exc())
            results['heatmap_path'] = ''
            
        # 10. 키워드 중심성 분석
        try:
            if network and len(network.nodes()) > 5:  # 최소한의 노드가 있는지 확인
                # 중심성 계산
                degree_centrality = nx.degree_centrality(network)
                betweenness_centrality = nx.betweenness_centrality(network, k=10)
                
                # 연결되지 않은 그래프에서도 작동하도록 예외 처리
                try:
                    eigenvector_centrality = nx.eigenvector_centrality_numpy(network)
                except Exception:
                    # 대체: 가장 큰 연결 요소에서만 고유벡터 중심성 계산
                    largest_cc = max(nx.connected_components(network), key=len)
                    subgraph = network.subgraph(largest_cc)
                    eigen_temp = nx.eigenvector_centrality_numpy(subgraph)
                    
                    # 전체 그래프에 결과 매핑
                    eigenvector_centrality = {}
                    for node in network:
                        if node in eigen_temp:
                            eigenvector_centrality[node] = eigen_temp[node]
                        else:
                            eigenvector_centrality[node] = 0.0
                
                # 상위 10개 노드 선택
                top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                top_nodes = [item[0] for item in top_degree]
                
                # 각 중심성 값 추출
                degree_vals = [degree_centrality[node] for node in top_nodes]
                betweenness_vals = [betweenness_centrality[node] for node in top_nodes]
                eigenvector_vals = [eigenvector_centrality[node] for node in top_nodes]
                
                # 그래프 생성
                centrality_path = os.path.join(STATIC_DIR, f'centrality_{unique_filename}.png')
                plt.figure(figsize=(12, 8))
                
                x = range(len(top_nodes))
                width = 0.25  # 막대 너비
                
                plt.bar([i - width for i in x], degree_vals, width, label='연결 중심성', color='skyblue')
                plt.bar(x, betweenness_vals, width, label='매개 중심성', color='lightgreen')
                plt.bar([i + width for i in x], eigenvector_vals, width, label='고유벡터 중심성', color='salmon')
                
                plt.xlabel('키워드', fontsize=12)
                plt.ylabel('중심성 값', fontsize=12)
                plt.title('키워드 중심성 분석 (상위 10개)', fontsize=16)
                plt.xticks(x, top_nodes, rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                plt.savefig(centrality_path, bbox_inches='tight')
                plt.close()
                
                if os.path.exists(centrality_path):
                    results['centrality_path'] = f'/static/centrality_{unique_filename}.png'
                else:
                    results['centrality_path'] = ''
            else:
                results['centrality_path'] = ''
        except Exception as centrality_error:
            logger.error(f"중심성 분석 오류: {centrality_error}")
            logger.error(traceback.format_exc())
            results['centrality_path'] = ''
        
        # 11. 클러스터링 분석 (고급 분석 탭)
        try:
            # 데이터가 충분한지 확인
            if analyzer.tf_idf_matrix is not None and analyzer.tf_idf_matrix.shape[0] > 5:
                logger.info(f"클러스터링 분석 시작: 문서 수 = {analyzer.tf_idf_matrix.shape[0]}")
                
                # 1) 클러스터링 시각화
                # TF-IDF 행렬을 2D로 차원 축소
                # 데이터 샘플 수에 따라 perplexity 조정
                n_samples = analyzer.tf_idf_matrix.shape[0]
                # perplexity는 보통 5~50 사이의 값 사용, 데이터 개수보다 작아야 함
                optimal_perplexity = min(30, max(5, n_samples // 3))
                
                # 안전하게 샘플 수보다 작은 값으로 설정
                if optimal_perplexity >= n_samples:
                    optimal_perplexity = max(2, n_samples - 1)
                    
                logger.info(f"2D t-SNE 설정: 샘플 수 = {n_samples}, perplexity = {optimal_perplexity}")
                tsne = TSNE(n_components=2, random_state=42, perplexity=optimal_perplexity)
                tsne_results = tsne.fit_transform(analyzer.tf_idf_matrix.toarray())
                
                # 클러스터링 수행 (K-means)
                num_clusters = min(5, analyzer.tf_idf_matrix.shape[0])  # 최대 5개 클러스터
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(analyzer.tf_idf_matrix.toarray())
                
                # 클러스터링 결과 시각화 경로
                clustering_path = os.path.join(STATIC_DIR, f'clustering_{unique_filename}.png')
                
                # 한글 폰트 설정 - 명시적으로 여기서 다시 설정
                try:
                    import matplotlib.font_manager as fm
                    # 가능한 한글 폰트 경로 목록
                    font_paths = [
                        'C:/Windows/Fonts/NanumGothic.ttf',
                        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                        '/app/fonts/NanumGothic.ttf'
                    ]
                    
                    font_set = False
                    for font_path in font_paths:
                        if os.path.exists(font_path):
                            plt.rcParams['font.family'] = 'NanumGothic'
                            plt.rcParams['axes.unicode_minus'] = False
                            logger.info(f"클러스터 분석용 한글 폰트 설정 완료: {font_path}")
                            font_set = True
                            break
                            
                    if not font_set:
                        logger.warning("클러스터 분석용 한글 폰트를 찾을 수 없습니다.")
                except Exception as font_err:
                    logger.error(f"클러스터 분석 폰트 설정 오류: {font_err}")
                
                # 클러스터링 결과 시각화
                plt.figure(figsize=(12, 8))
                
                colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))
                for i in range(num_clusters):
                    # 해당 클러스터에 속한 점 찾기
                    cluster_points = tsne_results[cluster_labels == i]
                    if len(cluster_points) > 0:  # 클러스터에 점이 있는지 확인
                        plt.scatter(
                            cluster_points[:, 0], 
                            cluster_points[:, 1], 
                            s=100, 
                            c=[colors[i]], 
                            label=f'클러스터 {i+1}'
                        )
                
                # 중심점 표시
                centers = kmeans.cluster_centers_
                if len(tsne_results) > 0:  # 결과가 있는지 확인
                    try:
                        # t-SNE를 사용하지 않고 클러스터링 된 포인트들의 평균 위치를 사용
                        centers_2d = np.zeros((num_clusters, 2))
                        
                        for i in range(num_clusters):
                            # 이 클러스터에 속한 점들의 평균 위치 계산
                            cluster_points = tsne_results[cluster_labels == i]
                            if len(cluster_points) > 0:
                                centers_2d[i] = np.mean(cluster_points, axis=0)
                        
                        # 중심점 시각화
                        plt.scatter(
                            centers_2d[:, 0], 
                            centers_2d[:, 1], 
                            s=200, 
                            c='black', 
                            alpha=0.5, 
                            marker='X'
                        )
                        logger.info("클러스터 중심점 시각화 성공")
                    except Exception as e:
                        logger.warning(f"클러스터 중심점 시각화 오류: {e}, 중심점 표시를 건너뜁니다.")
                        # 오류 발생 시 중심점 표시를 건너뜀
                
                plt.title('키워드 클러스터 분석', fontsize=16)
                plt.legend(fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(clustering_path, bbox_inches='tight')
                plt.close()
                
                # 파일 생성 확인
                if os.path.exists(clustering_path):
                    logger.info(f"클러스터링 분석 이미지 생성 완료: {clustering_path}")
                    results['clustering_path'] = f'/static/clustering_{unique_filename}.png'
                else:
                    logger.warning("클러스터링 분석 이미지 생성 실패")
                    results['clustering_path'] = ''
                
                # 클러스터 정보 생성
                community_info = []
                for i in range(num_clusters):
                    # 이 클러스터에 속한 문서 인덱스 찾기
                    cluster_docs = np.where(cluster_labels == i)[0]
                    
                    # 클러스터의 키워드 빈도 계산
                    cluster_word_freq = {}
                    for doc_idx in cluster_docs:
                        if doc_idx < len(analyzer.tokenized_corpus):
                            for word in analyzer.tokenized_corpus[doc_idx]:
                                if word in cluster_word_freq:
                                    cluster_word_freq[word] += 1
                                else:
                                    cluster_word_freq[word] = 1
                    
                    # 상위 키워드 추출
                    if cluster_word_freq:  # 단어가 있는지 확인
                        top_words = sorted(cluster_word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                        
                        community_info.append({
                            'id': i + 1,
                            'size': len(cluster_docs),
                            'top_words': [{'word': word, 'freq': freq} for word, freq in top_words]
                        })
                
                logger.info(f"커뮤니티 정보 생성 완료: {len(community_info)} 클러스터")
                results['community_info'] = community_info
                
                # 2) 키워드 영향력 버블 차트
                bubble_path = os.path.join(STATIC_DIR, f'bubble_{unique_filename}.png')
                
                try:
                    # 한글 폰트 설정 - 명시적으로 여기서 다시 설정
                    try:
                        import matplotlib.font_manager as fm
                        # 가능한 한글 폰트 경로 목록
                        font_paths = [
                            'C:/Windows/Fonts/NanumGothic.ttf',
                            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                            '/app/fonts/NanumGothic.ttf'
                        ]
                        
                        font_set = False
                        for font_path in font_paths:
                            if os.path.exists(font_path):
                                plt.rcParams['font.family'] = 'NanumGothic'
                                plt.rcParams['axes.unicode_minus'] = False
                                logger.info(f"버블차트용 한글 폰트 설정 완료: {font_path}")
                                font_set = True
                                break
                                
                        if not font_set:
                            logger.warning("버블차트용 한글 폰트를 찾을 수 없습니다.")
                    except Exception as font_err:
                        logger.error(f"버블차트 폰트 설정 오류: {font_err}")
                    
                    # 상위 30개 키워드 대상으로 버블 차트 생성
                    if hasattr(analyzer, 'word_freq') and analyzer.word_freq:
                        plt.figure(figsize=(12, 8))
                        
                        # 단어 빈도 데이터 가져오기
                        top_words = analyzer.word_freq.most_common(min(30, len(analyzer.word_freq)))
                        if top_words:  # 단어가 있는지 확인
                            word_list, freq_list = zip(*top_words)
                            
                            # 키워드 중요도 점수 계산 (TF-IDF 점수 평균)
                            importance_scores = []
                            for word in word_list:
                                if word in analyzer.tf_idf_feature_names:
                                    idx = list(analyzer.tf_idf_feature_names).index(word)
                                    score = np.mean(analyzer.tf_idf_matrix[:, idx].toarray())
                                    importance_scores.append(score)
                                else:
                                    importance_scores.append(0.01)  # 기본값
                            
                            # 버블 사이즈 계산
                            sizes = [f * 50 for f in freq_list]  # 빈도수에 비례
                            
                            # 색상 그라데이션 (중요도에 따라)
                            cmap = plt.cm.YlOrRd
                            norm = plt.Normalize(min(importance_scores), max(importance_scores))
                            colors = cmap(norm(importance_scores))
                            
                            # 버블 플롯 생성
                            fig, ax = plt.subplots(figsize=(12, 8))
                            scatter = ax.scatter(
                                range(len(word_list)), 
                                importance_scores, 
                                s=sizes, 
                                c=colors, 
                                alpha=0.7, 
                                edgecolors='gray'
                            )
                            
                            # 키워드 레이블 추가
                            for i, word in enumerate(word_list):
                                ax.annotate(
                                    word, 
                                    (i, importance_scores[i]),
                                    xytext=(0, 5),
                                    textcoords='offset points',
                                    ha='center', 
                                    fontsize=9
                                )
                            
                            # 컬러바 추가 (ax 매개변수 지정)
                            fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='중요도 점수')
                            ax.set_title('키워드 영향력 버블 차트', fontsize=16)
                            ax.set_xlabel('키워드', fontsize=12)
                            ax.set_ylabel('중요도 점수', fontsize=12)
                            ax.set_xticks(range(len(word_list)))
                            ax.set_xticklabels(word_list, rotation=90)
                            ax.grid(True, linestyle='--', alpha=0.7)
                            
                            plt.tight_layout()
                            plt.savefig(bubble_path, bbox_inches='tight')
                            plt.close()
                            
                            # 파일 생성 확인
                            if os.path.exists(bubble_path):
                                logger.info(f"버블 차트 이미지 생성 완료: {bubble_path}")
                                results['bubble_path'] = f'/static/bubble_{unique_filename}.png'
                            else:
                                logger.warning("버블 차트 이미지 생성 실패")
                                results['bubble_path'] = ''
                        else:
                            logger.warning("버블 차트 생성을 위한 단어 빈도 데이터가 없습니다.")
                            results['bubble_path'] = ''
                    else:
                        logger.warning("단어 빈도 정보가 없습니다.")
                        results['bubble_path'] = ''
                except Exception as bubble_error:
                    logger.error(f"버블 차트 생성 오류: {bubble_error}")
                    logger.error(traceback.format_exc())
                    results['bubble_path'] = ''
                
                # 3) 키워드 군집 3D 시각화
                clusters3d_path = os.path.join(STATIC_DIR, f'clusters3d_{unique_filename}.png')
                
                try:
                    # 한글 폰트 설정 - 명시적으로 여기서 다시 설정
                    try:
                        import matplotlib.font_manager as fm
                        # 가능한 한글 폰트 경로 목록
                        font_paths = [
                            'C:/Windows/Fonts/NanumGothic.ttf',
                            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                            '/app/fonts/NanumGothic.ttf'
                        ]
                        
                        font_set = False
                        for font_path in font_paths:
                            if os.path.exists(font_path):
                                plt.rcParams['font.family'] = 'NanumGothic'
                                plt.rcParams['axes.unicode_minus'] = False
                                logger.info(f"3D 시각화용 한글 폰트 설정 완료: {font_path}")
                                font_set = True
                                break
                                
                        if not font_set:
                            logger.warning("3D 시각화용 한글 폰트를 찾을 수 없습니다.")
                    except Exception as font_err:
                        logger.error(f"3D 시각화 폰트 설정 오류: {font_err}")
                    
                    # 3D 차원 축소 (TSNE)
                    if analyzer.tf_idf_matrix.shape[0] >= 4:  # 최소 4개 이상의 문서가 필요
                        # 데이터 샘플 수에 따라 perplexity 조정
                        n_samples = analyzer.tf_idf_matrix.shape[0]
                        # perplexity는 보통 5~50 사이의 값 사용, 데이터 개수보다 작아야 함
                        optimal_perplexity = min(30, max(5, n_samples // 3))
                        
                        # 안전하게 샘플 수보다 작은 값으로 설정
                        if optimal_perplexity >= n_samples:
                            optimal_perplexity = max(2, n_samples - 1)
                            
                        logger.info(f"3D t-SNE 설정: 샘플 수 = {n_samples}, perplexity = {optimal_perplexity}")
                        tsne_3d = TSNE(n_components=3, random_state=42, perplexity=optimal_perplexity)
                        tsne_results_3d = tsne_3d.fit_transform(analyzer.tf_idf_matrix.toarray())
                        
                        # 군집화 (Spectral Clustering)
                        n_clusters = min(4, analyzer.tf_idf_matrix.shape[0])
                        spectral = SpectralClustering(n_clusters=n_clusters, random_state=42, assign_labels='discretize')
                        spectral_labels = spectral.fit_predict(analyzer.tf_idf_matrix.toarray())
                        
                        # 3D 시각화
                        fig = plt.figure(figsize=(12, 10))
                        ax = fig.add_subplot(111, projection='3d')
                        
                        # 색상 생성
                        colors_3d = plt.cm.jet(np.linspace(0, 1, len(np.unique(spectral_labels))))
                        
                        # 각 클러스터 그리기
                        for i, label in enumerate(np.unique(spectral_labels)):
                            indices = spectral_labels == label
                            xs = tsne_results_3d[indices, 0]
                            ys = tsne_results_3d[indices, 1]
                            zs = tsne_results_3d[indices, 2]
                            if len(xs) > 0:  # 점이 있는지 확인
                                ax.scatter(xs, ys, zs, c=[colors_3d[i]], label=f'클러스터 {i+1}', s=50, alpha=0.8)
                        
                        ax.set_title('키워드 군집 3D 시각화', fontsize=16)
                        ax.view_init(35, 45)  # 시각화 각도 조정
                        ax.legend()
                        ax.grid(True)
                        plt.tight_layout()
                        plt.savefig(clusters3d_path, bbox_inches='tight', dpi=150)
                        plt.close()
                        
                        # 파일 생성 확인
                        if os.path.exists(clusters3d_path):
                            logger.info(f"3D 시각화 이미지 생성 완료: {clusters3d_path}")
                            results['clusters3d_path'] = f'/static/clusters3d_{unique_filename}.png'
                        else:
                            logger.warning("3D 시각화 이미지 생성 실패")
                            results['clusters3d_path'] = ''
                    else:
                        logger.warning("3D 시각화를 위한 충분한 데이터가 없습니다.")
                        results['clusters3d_path'] = ''
                except Exception as viz3d_error:
                    logger.error(f"3D 시각화 생성 오류: {viz3d_error}")
                    logger.error(traceback.format_exc())
                    results['clusters3d_path'] = ''
            else:
                logger.warning("고급 분석을 위한 충분한 데이터가 없습니다.")
                results['clustering_path'] = ''
                results['bubble_path'] = ''
                results['clusters3d_path'] = ''
                results['community_info'] = []
        except Exception as advanced_analysis_error:
            logger.error(f"고급 분석 오류: {advanced_analysis_error}")
            logger.error(traceback.format_exc())
            results['clustering_path'] = ''
            results['bubble_path'] = ''
            results['clusters3d_path'] = ''
            results['community_info'] = []
        
        # 결과 반환
        if background_tasks:
            # 백그라운드 작업으로 메모리 정리 및 임시 파일 삭제
            background_tasks.add_task(clean_memory)
            # 파일 삭제 지연 시간 추가 (30초 후)
            background_tasks.add_task(lambda fp=file_path: (asyncio.sleep(30) and os.remove(fp) if os.path.exists(fp) else None))
            # 추가 임시 파일 정리
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    background_tasks.add_task(lambda tf=temp_file: (asyncio.sleep(30) and os.remove(tf) if os.path.exists(tf) else None))
        
        # 최종 메모리 상태 출력
        logger.info("분석 완료 후 메모리 상태:")
        print_memory_info()
        
        return results
        
    except Exception as e:
        # 오류 발생 시 업로드된 파일이 있으면 삭제
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        # 상세한 에러 로그 출력
        logger.error(f"분석 중 오류가 발생했습니다: {str(e)}")
        logger.error(traceback.format_exc())
        clean_memory()  # 메모리 정리
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")

@app.post("/download_csv")
async def download_csv(data: List[dict]):
    try:
        # 임시 파일 경로를 애플리케이션 폴더 내에 생성
        temp_path = os.path.join(STATIC_DIR, f"temp_csv_{uuid.uuid4()}.csv")
            
        # CSV 파일 생성
        df = pd.DataFrame(data)
        df.to_csv(temp_path, index=False, encoding='utf-8-sig')  # UTF-8 BOM 인코딩 (Excel 호환)
        
        logger.info(f"CSV 파일 생성 완료: {temp_path}")
        
        # 백그라운드 작업으로 파일 삭제하는 함수 - 비동기 함수로 변경
        async def remove_file():
            try:
                # 잠시 대기 후 파일 삭제 (다운로드 완료 시간 고려)
                await asyncio.sleep(60)  # 60초 대기
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    logger.info(f"임시 CSV 파일 삭제 완료: {temp_path}")
            except Exception as e:
                logger.error(f"임시 파일 삭제 중 오류: {e}")
        
        # 파일 응답 생성
        return FileResponse(
            path=temp_path,
            filename="텍스트_분석_결과.csv",
            media_type="text/csv",
            background=remove_file
        )
    except Exception as e:
        logger.error(f"CSV 다운로드 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"CSV 파일 생성 중 오류: {str(e)}")

@app.post("/download_pdf")
async def download_zip(html_content: str = Form(...)):
    try:
        # 무거운 모듈이 아직 로드되지 않았다면 로드
        global heavy_modules_loaded
        if not heavy_modules_loaded:
            logger.info("HTML 다운로드 요청: 무거운 모듈 로드 시작")
            heavy_modules_loaded = load_heavy_modules()
            
        # 메모리 상태 확인
        memory_percent = get_memory_usage_percent()
        logger.info(f"HTML 다운로드 요청 - 현재 메모리 사용률: {memory_percent:.1f}%")
        
        # 임시 폴더 생성 (애플리케이션 폴더 내에)
        temp_dir_name = f"temp_zip_{uuid.uuid4()}"
        temp_dir = os.path.join(STATIC_DIR, temp_dir_name)
        os.makedirs(temp_dir, exist_ok=True)
        
        zip_path = os.path.join(temp_dir, "텍스트_분석_결과.zip")
        logger.info(f"ZIP 파일 생성 경로: {zip_path}")
        
        # HTML 파일에 필요한 스타일과 스크립트 추가
        enhanced_html = html_content
        
        # 1. CSS 스타일 추가 (HTML <head> 끝 부분에 추가)
        style_content = """
        <style>
            /* 섹션 간 여백 증가 */
            .result-section {
                margin-bottom: 50px;
                padding-bottom: 30px;
                clear: both;
            }
            
            /* 감정 분석 섹션 특별 여백 */
            #sentiment-content .result-section {
                margin-bottom: 70px;
            }
            
            /* 워드클라우드 섹션 */
            #sentimentCloudResults {
                padding-top: 60px;
                clear: both;
            }
            
            /* 감정 분석 워드클라우드 제목 */
            #sentiment-content .result-section:nth-child(2) h3 {
                margin-top: 50px;
                padding-top: 30px;
            }
            
            /* 프로그레스 바 아래 여백 */
            .progress {
                margin-bottom: 50px;
            }
        </style>
        """
        
        if '</head>' in enhanced_html:
            enhanced_html = enhanced_html.replace('</head>', f'{style_content}</head>')
        else:
            # HTML에 <head> 태그가 없을 경우 추가
            enhanced_html = f'<head>{style_content}</head>{enhanced_html}'
        
        # 2. HTML 직접 수정 - 탭 메뉴와 다운로드 버튼 숨기기
        # 탭 메뉴(resultTabs) 숨기기
        import re
        
        # 탭 메뉴 부분을 찾아서 스타일 속성 추가
        tab_pattern = r'<ul\s+class="nav\s+nav-tabs"\s+id="resultTabs".*?>(.*?)</ul>'
        enhanced_html = re.sub(tab_pattern, 
                              r'<ul class="nav nav-tabs" id="resultTabs" style="display:none;">\1</ul>', 
                              enhanced_html, 
                              flags=re.DOTALL)
        
        # 다운로드 버튼 부분 숨기기
        download_pattern = r'<div\s+class="d-flex\s+justify-content-center\s+gap-3\s+mt-4\s+mb-5">(.*?)</div>'
        enhanced_html = re.sub(download_pattern, 
                              r'<div class="d-flex justify-content-center gap-3 mt-4 mb-5" style="display:none;">\1</div>', 
                              enhanced_html, 
                              flags=re.DOTALL)
        
        # 모든 탭 콘텐츠를 활성화 (fade 클래스 제거, show 및 active 클래스 추가)
        tab_pane_pattern = r'<div\s+class="tab-pane\s+fade(?:\s+animated)?(?:\s+show)?(?:\s+active)?"\s+id="([^"]+)"'
        enhanced_html = re.sub(tab_pane_pattern, 
                              r'<div class="tab-pane show active" id="\1"', 
                              enhanced_html)
        
        # 3. JavaScript 추가 (HTML <body> 끝 부분에 추가) - 보험으로 남겨둠
        script_content = """
        <script>
            // 페이지 로드 시 실행
            document.addEventListener('DOMContentLoaded', function() {
                // 탭 버튼들을 숨기기 (상단 탭 메뉴 전체)
                const tabsNav = document.querySelector('#resultTabs');
                if (tabsNav) {
                    tabsNav.style.display = 'none';
                }
                
                // 모든 탭 콘텐츠를 표시 (숨겨진 콘텐츠도 모두 표시)
                document.querySelectorAll('.tab-pane').forEach(pane => {
                    pane.classList.add('show', 'active');
                    pane.classList.remove('fade'); // 페이드 효과 제거
                });
                
                // 다운로드 버튼들이 있는 div 전체를 제거
                const downloadButtonsDiv = document.querySelector('.d-flex.justify-content-center.gap-3.mt-4.mb-5');
                if (downloadButtonsDiv) {
                    downloadButtonsDiv.style.display = 'none'; // 완전히 숨기기
                }
            });
        </script>
        """
        
        if '</body>' in enhanced_html:
            enhanced_html = enhanced_html.replace('</body>', f'{script_content}</body>')
        else:
            # HTML에 <body> 태그가 없을 경우 추가
            enhanced_html = f'{enhanced_html}<script>{script_content}</script>'
        
        # 수정된 HTML 저장
        html_path = os.path.join(temp_dir, "분석_결과.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(enhanced_html)
        
        logger.info(f"HTML 파일 저장 완료: {html_path}")
        
        # 이미지 파일 복사 (from static 폴더)
        # 이미지 파일 경로 추출
        img_paths = re.findall(r'src="(/static/[^"]+)"', enhanced_html)
        logger.info(f"발견된 이미지 파일 수: {len(img_paths)}")
        
        for img_path in img_paths:
            # 상대 경로를 절대 경로로 변환
            full_path = os.path.join(BASE_DIR, img_path.lstrip('/'))
            # 대상 경로
            target_path = os.path.join(temp_dir, os.path.basename(img_path))
            # 파일 복사
            if os.path.exists(full_path):
                shutil.copy2(full_path, target_path)
                logger.info(f"이미지 파일 복사: {full_path} → {target_path}")
                
                # HTML 파일 내 이미지 경로 수정
                enhanced_html = enhanced_html.replace(
                    f'src="{img_path}"', 
                    f'src="{os.path.basename(img_path)}"'
                )
        
        # 수정된 HTML 다시 저장
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(enhanced_html)
        
        # ZIP 파일 생성
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            # HTML 파일 추가
            zip_file.write(html_path, "분석_결과.html")
            
            # 이미지 파일 추가
            for img_path in img_paths:
                full_path = os.path.join(BASE_DIR, img_path.lstrip('/'))
                if os.path.exists(full_path):
                    zip_file.write(full_path, os.path.basename(img_path))
        
        logger.info(f"ZIP 파일 생성 완료: {zip_path}")
        
        # 백그라운드 작업으로 임시 폴더 삭제하는 함수 - 비동기 함수로 변경
        async def remove_temp_dir():
            try:
                # 잠시 대기 후 파일 삭제 (다운로드 완료 시간 고려)
                await asyncio.sleep(60)  # 60초 대기
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"임시 폴더 삭제 완료: {temp_dir}")
            except Exception as e:
                logger.error(f"임시 폴더 삭제 중 오류: {e}")
        
        # 파일 응답 생성
        return FileResponse(
            path=zip_path,
            filename="텍스트_분석_결과.zip",
            media_type="application/zip",
            background=remove_temp_dir
        )
    except Exception as e:
        logger.error(f"ZIP 다운로드 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"ZIP 파일 생성 중 오류: {str(e)}")

# 서버 시작 중 에러 디버깅을 위한 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    """서버 상태 확인을 위한 헬스체크 엔드포인트"""
    memory_info = print_memory_info()
    return {
        "status": "healthy",
        "memory_usage_mb": memory_info,
        "memory_usage_percent": get_memory_usage_percent(),
        "memory_limit_mb": MEMORY_LIMIT_MB,
        "python_version": sys.version,
        "heavy_modules_loaded": heavy_modules_loaded
    }

if __name__ == "__main__":
    # 로컬 개발 시 사용
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True) 