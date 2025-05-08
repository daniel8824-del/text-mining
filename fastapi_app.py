import sys
import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request

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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "pos_options": POS_OPTIONS})

@app.post("/analyze")
async def analyze_text(
    file: UploadFile = File(...),
    text_column: str = Form(...),
    pos_tags: Optional[List[str]] = Form(None),
    stopwords: Optional[str] = Form(None)
):
    # 파일 유효성 검사
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV 파일만 지원합니다.")

    try:
        # 고유한 파일명 생성 (UUID 사용)
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # 파일 저장
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 사용자 정의 불용어 처리
        custom_stopwords = []
        if stopwords:
            custom_stopwords = [word.strip() for word in stopwords.split(',') if word.strip()]
        
        # 텍스트 마이닝 분석 실행
        analyzer = TextMiningAnalysis(file_path=file_path, text_column=text_column)
        
        # 품사 태깅 및 전처리
        pos_filter = pos_tags if pos_tags else None
        analyzer.preprocess_text(pos_filter=pos_filter, custom_stopwords=custom_stopwords)
        
        # 분석 결과 얻기
        results = {}
        
        # 1. 키워드 추출
        keywords = analyzer.extract_keywords(top_n=20)
        results['keywords'] = [{'word': word, 'freq': freq} for word, freq in keywords.items()]
        
        # 2. TF-IDF 분석
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
        
        # 3. 토픽 모델링
        topics = analyzer.topic_modeling(num_topics=5, num_words=10)
        results['topics'] = []
        for i, topic in enumerate(topics):
            topic_words = [{'word': word, 'score': float(score)} for word, score in topic]
            results['topics'].append({
                'topic_id': i + 1,
                'words': topic_words
            })
        
        # 4. 감정 분석
        try:
            sentiment = analyzer.sentiment_analysis()
            if isinstance(sentiment, pd.DataFrame):
                sentiment_counts = sentiment['Sentiment'].value_counts()
                results['sentiment'] = {
                    'positive': int(sentiment_counts.get('Positive', 0)),
                    'negative': int(sentiment_counts.get('Negative', 0)),
                    'neutral': int(sentiment_counts.get('Neutral', 0))
                }
            else:
                # 감정 분석 결과가 DataFrame이 아닌 경우 기본값 설정
                results['sentiment'] = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
        except Exception as sentiment_error:
            print(f"감정 분석 오류: {sentiment_error}")
            results['sentiment'] = {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
        
        # 5. 워드클라우드 생성
        try:
            # 폰트 경로 설정 (운영체제별 처리)
            font_path = None
            if os.path.exists('C:/Windows/Fonts/malgun.ttf'):  # Windows
                font_path = 'C:/Windows/Fonts/malgun.ttf'
            elif os.path.exists('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'):  # Ubuntu with Nanum
                font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
            
            wordcloud = analyzer.create_word_cloud(font_path=font_path)
            wordcloud_path = os.path.join(STATIC_DIR, f'wordcloud_{unique_filename}.png')
            
            # 워드클라우드 시각화 (제목 추가)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title('전체 키워드 워드클라우드', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(wordcloud_path, bbox_inches='tight')
            plt.close()
            
            results['wordcloud_path'] = f'/static/wordcloud_{unique_filename}.png'
        except Exception as wordcloud_error:
            print(f"워드클라우드 생성 오류: {wordcloud_error}")
            results['wordcloud_path'] = ''
            
        # 6. 키워드 네트워크 분석
        try:
            network = analyzer.keyword_network_analysis(threshold=2, top_n=30)
            if network:
                # 네트워크 시각화
                plt.figure(figsize=(10, 8))
                analyzer.plot_network(network)
                
                # 네트워크 그래프 저장
                network_path = os.path.join(STATIC_DIR, f'network_{unique_filename}.png')
                plt.savefig(network_path, bbox_inches='tight')
                plt.close()
                
                results['network_path'] = f'/static/network_{unique_filename}.png'
                
                # 네트워크 노드 정보 추가
                node_data = []
                for node in network.nodes():
                    node_data.append({
                        'word': node,
                        'size': network.nodes[node]['size']
                    })
                results['network_nodes'] = sorted(node_data, key=lambda x: x['size'], reverse=True)[:20]
        except Exception as network_error:
            print(f"키워드 네트워크 분석 오류: {network_error}")
            results['network_path'] = ''
            results['network_nodes'] = []
        
        # 파일 처리 완료 후 삭제 (선택적)
        os.remove(file_path)
        
        return results
        
    except Exception as e:
        # 오류 발생 시 업로드된 파일이 있으면 삭제
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")

@app.post("/download_csv")
async def download_csv(data: List[dict]):
    try:
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            temp_path = tmp_file.name
            
        # CSV 파일 생성
        df = pd.DataFrame(data)
        df.to_csv(temp_path, index=False, encoding='utf-8-sig')  # UTF-8 BOM 인코딩 (Excel 호환)
        
        # 파일 응답 생성
        return FileResponse(
            path=temp_path,
            filename="텍스트_분석_결과.csv",
            media_type="text/csv",
            background=lambda: os.remove(temp_path)  # 다운로드 후 임시 파일 삭제
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV 파일 생성 중 오류: {str(e)}")

@app.post("/download_pdf")
async def download_zip(html_content: str = Form(...)):
    try:
        # 임시 폴더 생성
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "텍스트_분석_결과.zip")
        
        # 정적 이미지 파일 복사
        image_files = []
        for filename in os.listdir(STATIC_DIR):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(STATIC_DIR, filename)
                image_files.append((image_path, filename))
        
        # HTML 파일 생성
        html_path = os.path.join(temp_dir, "분석결과.html")
        with open(html_path, "w", encoding="utf-8") as f:
            # 기본 HTML 구조 추가 (CSS 포함)
            html_content_with_header = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>텍스트 마이닝 분석 결과</title>
                <style>
                    body {{ font-family: 'Arial', sans-serif; }}
                    .result-section {{ 
                        margin: 20px 0; 
                        padding: 15px; 
                        border: 1px solid #ddd; 
                        border-radius: 5px; 
                    }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ 
                        width: 100%; 
                        border-collapse: collapse; 
                        margin: 15px 0; 
                    }}
                    th, td {{ 
                        padding: 8px; 
                        border: 1px solid #ddd; 
                        text-align: left; 
                    }}
                    th {{ background-color: #f2f2f2; }}
                    img {{ max-width: 100%; height: auto; }}
                </style>
            </head>
            <body>
                <h1>텍스트 마이닝 분석 결과</h1>
                {html_content}
            </body>
            </html>
            """
            f.write(html_content_with_header)
        
        # ZIP 파일 생성
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            # HTML 파일 추가
            zip_file.write(html_path, arcname="분석결과.html")
            
            # 이미지 파일 추가
            for image_path, filename in image_files:
                try:
                    if os.path.exists(image_path):
                        zip_file.write(image_path, arcname=f"images/{filename}")
                except Exception as img_error:
                    print(f"이미지 파일 처리 오류: {img_error}")
        
        # 파일 응답 생성
        return FileResponse(
            path=zip_path,
            filename="텍스트_분석_결과.zip",
            media_type="application/zip",
            background=lambda: shutil.rmtree(temp_dir)  # 다운로드 후 임시 폴더 삭제
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ZIP 파일 생성 중 오류: {str(e)}")

# 앱 실행
if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True) 