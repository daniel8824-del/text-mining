from flask import Flask, render_template, request, jsonify, send_file, Response
from text_mining_analysis import TextMiningAnalysis
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지 파일만 생성하는 백엔드
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import networkx as nx
from wordcloud import WordCloud
import io
import csv
import json
import zipfile
import re

app = Flask(__name__)

# 현재 디렉토리 경로를 기준으로 필요한 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STOPWORDS_PATH = os.path.join(BASE_DIR, 'korean_stopwords.txt')
SENTIMENT_DICT_PATH = os.path.join(BASE_DIR, 'knu_sentiment_lexicon.csv')

# 품사 태깅 옵션
POS_OPTIONS = {
    'Noun': '명사',
    'Verb': '동사',
    'Adjective': '형용사',
    'Adverb': '부사',
    'Determiner': '관형사'
}

@app.route('/')
def index():
    return render_template('index.html', pos_options=POS_OPTIONS)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # 파일 업로드 처리
        if 'file' not in request.files:
            return jsonify({'error': '파일이 업로드되지 않았습니다.'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
        # 파일 저장
        upload_dir = os.path.join(BASE_DIR, 'data', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        
        # 분석 옵션 가져오기
        text_column = request.form.get('text_column', '')
        selected_pos = request.form.getlist('pos_tags')
        stopwords = request.form.get('stopwords', '').split(',')
        stopwords = [word.strip() for word in stopwords if word.strip()]
        
        # 텍스트 마이닝 분석 실행
        analyzer = TextMiningAnalysis(file_path=file_path, text_column=text_column)
        
        # 품사 태깅 및 전처리
        pos_filter = selected_pos if selected_pos else None
        analyzer.preprocess_text(pos_filter=pos_filter, custom_stopwords=stopwords)
        
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
            wordcloud_path = os.path.join(BASE_DIR, 'static', 'wordcloud.png')
            os.makedirs(os.path.join(BASE_DIR, 'static'), exist_ok=True)
            wordcloud.to_file(wordcloud_path)
            
            # 워드클라우드 시각화 (제목 추가)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title('전체 키워드 워드클라우드', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(wordcloud_path, bbox_inches='tight')
            plt.close()
            
            results['wordcloud_path'] = '/static/wordcloud.png'  # URL 형식으로 경로 지정
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
                network_path = os.path.join(BASE_DIR, 'static', 'network.png')
                plt.savefig(network_path, bbox_inches='tight')
                plt.close()
                
                results['network_path'] = '/static/network.png'
                
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
        
        # 7. 키워드 상관관계 히트맵
        try:
            # 키워드 동시 출현 빈도로 상관관계 계산
            if len(analyzer.tokenized_corpus) > 0:
                # 상위 키워드 추출
                all_words = [word for doc in analyzer.tokenized_corpus for word in doc]
                word_counts = Counter(all_words)
                top_keywords = [word for word, _ in word_counts.most_common(15)]  # 상위 15개 키워드
                
                # 상관관계 행렬 생성
                correlation_matrix = np.zeros((len(top_keywords), len(top_keywords)))
                
                # 동시 출현 빈도 계산
                for i, word1 in enumerate(top_keywords):
                    for j, word2 in enumerate(top_keywords):
                        if i == j:
                            correlation_matrix[i][j] = 1.0  # 대각선은 1로 설정
                        else:
                            # 두 단어가 동시에 나타나는 문서 수 계산
                            cooccurrence = sum(1 for doc in analyzer.tokenized_corpus 
                                            if word1 in doc and word2 in doc)
                            # 정규화
                            doc_with_word1 = sum(1 for doc in analyzer.tokenized_corpus if word1 in doc)
                            doc_with_word2 = sum(1 for doc in analyzer.tokenized_corpus if word2 in doc)
                            
                            if doc_with_word1 > 0 and doc_with_word2 > 0:
                                correlation = cooccurrence / (doc_with_word1 * doc_with_word2) ** 0.5
                                correlation_matrix[i][j] = correlation
                
                # 히트맵 생성
                plt.figure(figsize=(10, 8))
                
                # 한글 폰트 설정
                plt.rcParams['font.family'] = 'Malgun Gothic'
                plt.rcParams['axes.unicode_minus'] = False
                
                sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu',
                            xticklabels=top_keywords, yticklabels=top_keywords, fmt='.2f')
                plt.title('키워드 상관관계', fontsize=16, fontfamily='Malgun Gothic')
                
                # 히트맵 저장
                heatmap_path = os.path.join(BASE_DIR, 'static', 'heatmap.png')
                plt.savefig(heatmap_path, bbox_inches='tight')
                plt.close()
                
                results['heatmap_path'] = '/static/heatmap.png'
                results['correlation_keywords'] = top_keywords
        except Exception as heatmap_error:
            print(f"상관관계 히트맵 생성 오류: {heatmap_error}")
            results['heatmap_path'] = ''
            results['correlation_keywords'] = []
            
        # 8. 긍정/부정 워드클라우드
        try:
            # 감정 분석을 통해 긍정/부정 단어 빈도 계산
            pos_words = {}
            neg_words = {}
            
            for doc in analyzer.tokenized_corpus:
                # 단어별 감정 점수 계산
                for word in doc:
                    sentiment_score = analyzer.knu_dict.get(word, 0)
                    if sentiment_score > 0:  # 긍정 단어
                        pos_words[word] = pos_words.get(word, 0) + 1
                    elif sentiment_score < 0:  # 부정 단어
                        neg_words[word] = neg_words.get(word, 0) + 1
            
            # 긍정 워드클라우드 생성
            if pos_words:
                pos_words_text = ' '.join([f"{word} " * count for word, count in pos_words.items()])
                pos_wordcloud = WordCloud(
                    font_path='C:/Windows/Fonts/malgun.ttf',
                    width=800, 
                    height=400, 
                    background_color='white',
                    max_words=100,
                    colormap='YlGn'  # 녹색 계열 색상
                ).generate(pos_words_text)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(pos_wordcloud, interpolation='bilinear')
                plt.title('긍정 단어 워드클라우드', fontsize=16, fontfamily='Malgun Gothic')
                plt.axis('off')
                plt.tight_layout()
                
                pos_cloud_path = os.path.join(BASE_DIR, 'static', 'wordcloud_positive.png')
                plt.savefig(pos_cloud_path, bbox_inches='tight')
                plt.close()
                
                results['pos_wordcloud_path'] = '/static/wordcloud_positive.png'
            
            # 부정 워드클라우드 생성
            if neg_words:
                neg_words_text = ' '.join([f"{word} " * count for word, count in neg_words.items()])
                neg_wordcloud = WordCloud(
                    font_path='C:/Windows/Fonts/malgun.ttf',
                    width=800, 
                    height=400, 
                    background_color='white',
                    max_words=100,
                    colormap='OrRd'  # 붉은 계열 색상
                ).generate(neg_words_text)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(neg_wordcloud, interpolation='bilinear')
                plt.title('부정 단어 워드클라우드', fontsize=16, fontfamily='Malgun Gothic')
                plt.axis('off')
                plt.tight_layout()
                
                neg_cloud_path = os.path.join(BASE_DIR, 'static', 'wordcloud_negative.png')
                plt.savefig(neg_cloud_path, bbox_inches='tight')
                plt.close()
                
                results['neg_wordcloud_path'] = '/static/wordcloud_negative.png'
        except Exception as sentiment_cloud_error:
            print(f"감정 워드클라우드 생성 오류: {sentiment_cloud_error}")
            results['pos_wordcloud_path'] = ''
            results['neg_wordcloud_path'] = ''
            
        # 9. 토픽 점유율 차트
        try:
            if hasattr(analyzer, 'lda_model') and analyzer.lda_model is not None:
                # 토픽 분포 계산
                topic_distribution = analyzer.lda_model.transform(analyzer.tf_idf_matrix)
                topic_shares = topic_distribution.mean(axis=0)
                
                # 토픽별 색상 설정
                colors = plt.cm.tab10(np.arange(len(topic_shares)))
                
                # 토픽 이름 생성
                topic_names = [f"토픽 {i+1}" for i in range(len(topic_shares))]
                
                # 한글 폰트 설정
                plt.rcParams['font.family'] = 'Malgun Gothic'
                plt.rcParams['axes.unicode_minus'] = False
                
                # 파이 차트 생성
                plt.figure(figsize=(10, 8))
                plt.pie(topic_shares, labels=topic_names, autopct='%1.1f%%', 
                        colors=colors, startangle=90, shadow=True, textprops={'fontsize': 12, 'fontfamily': 'Malgun Gothic'})
                plt.title('토픽 점유율', fontsize=16, fontfamily='Malgun Gothic')
                plt.axis('equal')
                
                # 파이 차트 저장
                topic_shares_path = os.path.join(BASE_DIR, 'static', 'topic_shares.png')
                plt.savefig(topic_shares_path, bbox_inches='tight')
                plt.close()
                
                results['topic_shares_path'] = '/static/topic_shares.png'
                results['topic_shares_data'] = [float(share) for share in topic_shares]
        except Exception as topic_error:
            print(f"토픽 점유율 차트 생성 오류: {topic_error}")
            results['topic_shares_path'] = ''
            results['topic_shares_data'] = []
            
        # 10. 중심성 분석 차트
        try:
            if 'network_path' in results and results['network_path']:
                # 이미 네트워크가 생성되어 있으므로 중심성 분석 수행
                network = analyzer.keyword_network_analysis(threshold=2, top_n=30)
                
                # 중심성 계산
                degree_centrality = nx.degree_centrality(network)
                betweenness_centrality = nx.betweenness_centrality(network, k=10)
                
                try:
                    eigenvector_centrality = nx.eigenvector_centrality_numpy(network)
                except:
                    # 대체: 연결 중심성 사용
                    eigenvector_centrality = degree_centrality
                
                # 상위 10개 노드만 선택
                top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                top_nodes = [item[0] for item in top_degree]
                
                # 각 중심성 값 추출
                degree_vals = [degree_centrality[node] for node in top_nodes]
                betweenness_vals = [betweenness_centrality[node] for node in top_nodes]
                eigenvector_vals = [eigenvector_centrality[node] for node in top_nodes]
                
                # 한글 폰트 설정
                plt.rcParams['font.family'] = 'Malgun Gothic'
                plt.rcParams['axes.unicode_minus'] = False
                
                # 막대 그래프 생성
                plt.figure(figsize=(12, 8))
                x = range(len(top_nodes))
                width = 0.25  # 막대 너비
                
                plt.bar([i - width for i in x], degree_vals, width, label='연결 중심성', color='skyblue')
                plt.bar(x, betweenness_vals, width, label='매개 중심성', color='lightgreen')
                plt.bar([i + width for i in x], eigenvector_vals, width, label='고유벡터 중심성', color='salmon')
                
                plt.xlabel('키워드', fontsize=12, fontfamily='Malgun Gothic')
                plt.ylabel('중심성 값', fontsize=12, fontfamily='Malgun Gothic')
                plt.title('키워드 중심성 분석 (상위 10개)', fontsize=16, fontfamily='Malgun Gothic')
                plt.xticks(x, top_nodes, rotation=45, ha='right', fontfamily='Malgun Gothic')
                plt.legend(prop={'family': 'Malgun Gothic'})
                plt.tight_layout()
                
                # 차트 저장
                centrality_path = os.path.join(BASE_DIR, 'static', 'centrality_analysis.png')
                plt.savefig(centrality_path, bbox_inches='tight')
                plt.close()
                
                results['centrality_path'] = '/static/centrality_analysis.png'
                results['top_nodes'] = top_nodes
                results['centrality_data'] = {
                    'degree': degree_vals,
                    'betweenness': betweenness_vals,
                    'eigenvector': eigenvector_vals
                }
        except Exception as centrality_error:
            print(f"중심성 분석 차트 생성 오류: {centrality_error}")
            results['centrality_path'] = ''
            
        # 11. 클러스터링 분석
        try:
            if 'network_path' in results and results['network_path']:
                network = analyzer.keyword_network_analysis(threshold=2, top_n=30)
                
                # 커뮤니티 탐지
                communities = list(nx.algorithms.community.greedy_modularity_communities(network))
                
                # 노드별 커뮤니티 할당
                node_community = {}
                for i, community in enumerate(communities):
                    for node in community:
                        node_community[node] = i
                
                # 노드 크기와 엣지 두께 설정
                node_sizes = [network.nodes[node]['size'] * 30 for node in network.nodes()]
                edge_weights = [network.edges[edge]['weight'] for edge in network.edges()]
                
                # 각 커뮤니티별 색상 설정
                community_colors = [plt.cm.tab20(i) for i in range(len(communities))]
                node_colors = [community_colors[node_community.get(node, 0)] for node in network.nodes()]
                
                # 레이아웃 설정
                pos = nx.spring_layout(network, k=0.3, seed=42)
                
                # 한글 폰트 설정
                plt.rcParams['font.family'] = 'Malgun Gothic'
                plt.rcParams['axes.unicode_minus'] = False
                
                # 그래프 그리기
                plt.figure(figsize=(15, 15))
                nx.draw_networkx_nodes(network, pos, node_size=node_sizes, 
                                    node_color=node_colors, alpha=0.8)
                nx.draw_networkx_edges(network, pos, width=edge_weights, alpha=0.3)
                nx.draw_networkx_labels(network, pos, font_size=10, font_family='Malgun Gothic')
                
                # 커뮤니티 정보 추가
                for i, community in enumerate(communities[:5]):  # 상위 5개 커뮤니티만 라벨 표시
                    if community:
                        # 커뮤니티 중심 계산
                        comm_center_x = sum(pos[node][0] for node in community) / len(community)
                        comm_center_y = sum(pos[node][1] for node in community) / len(community)
                        
                        # 커뮤니티 주요 키워드 추출 (상위 3개)
                        top_words = sorted([(node, network.nodes[node]['size']) for node in community], 
                                        key=lambda x: x[1], reverse=True)[:3]
                        comm_label = f"클러스터 {i+1}\n" + ", ".join([word for word, _ in top_words])
                        
                        plt.text(comm_center_x, comm_center_y, comm_label, 
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                                horizontalalignment='center', fontsize=12, family='Malgun Gothic')
                
                plt.axis('off')
                plt.title('키워드 클러스터 분석', fontsize=16, fontfamily='Malgun Gothic')
                
                # 그래프 저장
                clustering_path = os.path.join(BASE_DIR, 'static', 'clustering_analysis.png')
                plt.savefig(clustering_path, bbox_inches='tight')
                plt.close()
                
                results['clustering_path'] = '/static/clustering_analysis.png'
                
                # 커뮤니티 정보 추가
                community_info = []
                for i, community in enumerate(communities[:5]):
                    if community:
                        top_words = sorted([(node, network.nodes[node]['size']) for node in community], 
                                        key=lambda x: x[1], reverse=True)[:5]
                        community_info.append({
                            'id': i + 1,
                            'size': len(community),
                            'top_words': [{'word': word, 'size': size} for word, size in top_words]
                        })
                
                results['community_info'] = community_info
        except Exception as clustering_error:
            print(f"클러스터링 분석 생성 오류: {clustering_error}")
            results['clustering_path'] = ''
            
        # 12. 키워드 영향력 버블 차트
        try:
            if 'centrality_path' in results and results['centrality_path']:
                network = analyzer.keyword_network_analysis(threshold=2, top_n=30)
                
                # 중심성 계산 (이미 위에서 계산됨)
                degree_centrality = nx.degree_centrality(network)
                betweenness_centrality = nx.betweenness_centrality(network, k=10)
                
                try:
                    eigenvector_centrality = nx.eigenvector_centrality_numpy(network)
                except:
                    eigenvector_centrality = degree_centrality
                
                # 상위 20개 노드 선택
                top_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
                nodes = [item[0] for item in top_centrality]
                
                # 각 노드의 크기, 색상 결정
                sizes = [network.nodes[node]['size'] * 100 for node in nodes]
                colors = [eigenvector_centrality[node] for node in nodes]
                
                # 버블 차트 생성
                plt.figure(figsize=(12, 8))
                
                # 한글 폰트 설정
                plt.rcParams['font.family'] = 'Malgun Gothic'
                plt.rcParams['axes.unicode_minus'] = False
                
                x = [degree_centrality[node] for node in nodes]  # 연결 중심성
                y = [betweenness_centrality[node] for node in nodes]  # 매개 중심성
                
                scatter = plt.scatter(x, y, s=sizes, c=colors, alpha=0.7, cmap='viridis', 
                                    edgecolors='gray', linewidths=1)
                
                # 각 노드에 라벨 추가
                for i, node in enumerate(nodes):
                    plt.annotate(node, (x[i], y[i]), 
                                xytext=(5, 5), textcoords='offset points', 
                                fontsize=9, alpha=0.8, fontfamily='Malgun Gothic')
                
                plt.colorbar(scatter, label='고유벡터 중심성')
                plt.xlabel('연결 중심성', fontsize=12, fontfamily='Malgun Gothic')
                plt.ylabel('매개 중심성', fontsize=12, fontfamily='Malgun Gothic')
                plt.title('키워드 영향력 버블 차트', fontsize=16, fontfamily='Malgun Gothic')
                
                # 차트 저장
                bubble_path = os.path.join(BASE_DIR, 'static', 'keyword_influence_bubble.png')
                plt.savefig(bubble_path, bbox_inches='tight')
                plt.close()
                
                results['bubble_path'] = '/static/keyword_influence_bubble.png'
                
                # 버블 차트 데이터
                bubble_data = []
                for i, node in enumerate(nodes):
                    bubble_data.append({
                        'word': node,
                        'x': float(x[i]),  # 연결 중심성
                        'y': float(y[i]),  # 매개 중심성
                        'size': int(sizes[i] / 100),  # 크기 (빈도수)
                        'color': float(colors[i])  # 색상 (고유벡터 중심성)
                    })
                
                results['bubble_data'] = bubble_data
        except Exception as bubble_error:
            print(f"키워드 영향력 버블 차트 생성 오류: {bubble_error}")
            results['bubble_path'] = ''
            
        # 13. 키워드 군집 3D 시각화
        try:
            from mpl_toolkits.mplot3d import Axes3D
            from sklearn.manifold import TSNE
            
            if len(analyzer.tokenized_corpus) > 0 and hasattr(analyzer, 'tf_idf_matrix'):
                # 상위 100개 키워드 선택
                all_words = [word for doc in analyzer.tokenized_corpus for word in doc]
                word_counts = Counter(all_words)
                top_words = [word for word, _ in word_counts.most_common(100)]
                
                # 단어 벡터 생성을 위한 인덱스 찾기
                word_indices = [list(analyzer.tf_idf_feature_names).index(word) 
                                for word in top_words if word in analyzer.tf_idf_feature_names]
                
                if word_indices:
                    # TF-IDF 행렬에서 해당 단어들의 벡터 추출
                    word_vectors = np.zeros((len(word_indices), analyzer.tf_idf_matrix.shape[0]))
                    for i, idx in enumerate(word_indices):
                        word_vectors[i] = analyzer.tf_idf_matrix[:, idx].toarray().flatten()
                    
                    # t-SNE로 3차원 축소
                    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, max(5, len(word_indices)//5)))
                    word_vectors_3d = tsne.fit_transform(word_vectors)
                    
                    # 3D 시각화
                    fig = plt.figure(figsize=(12, 10))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # 한글 폰트 설정
                    plt.rcParams['font.family'] = 'Malgun Gothic'
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    # 커뮤니티 정보 가져오기 (이전 클러스터링 분석에서 계산됨)
                    if 'community_info' in results:
                        # 네트워크 재생성 및 커뮤니티 탐지
                        network = analyzer.keyword_network_analysis(threshold=2, top_n=50)
                        communities = list(nx.algorithms.community.greedy_modularity_communities(network))
                        
                        # 단어별 커뮤니티 매핑
                        word_to_community = {}
                        for i, community in enumerate(communities):
                            for word in community:
                                word_to_community[word] = i
                        
                        # 각 단어의 색상 및 크기 설정
                        colors = []
                        sizes = []
                        visible_words = []
                        visible_coords = []
                        
                        for i, word in enumerate(top_words):
                            if i < len(word_indices):
                                community_id = word_to_community.get(word, 0)
                                colors.append(community_id)
                                sizes.append(word_counts.get(word, 1) * 20)
                                visible_words.append(word)
                                visible_coords.append(word_vectors_3d[i])
                        
                        if visible_coords:
                            visible_coords = np.array(visible_coords)
                            scatter = ax.scatter(
                                visible_coords[:, 0], visible_coords[:, 1], visible_coords[:, 2],
                                c=colors, s=sizes, alpha=0.7, cmap='tab20'
                            )
                            
                            # 주요 키워드에만 라벨 추가
                            threshold = np.percentile([word_counts.get(w, 0) for w in visible_words], 70)
                            for i, word in enumerate(visible_words):
                                if word_counts.get(word, 0) > threshold:
                                    ax.text(visible_coords[i, 0], visible_coords[i, 1], visible_coords[i, 2], 
                                          word, fontsize=9, fontfamily='Malgun Gothic')
                            
                            ax.set_title('키워드 3D 군집 시각화', fontsize=16, fontfamily='Malgun Gothic')
                            plt.tight_layout()
                            
                            # 3D 클러스터 저장
                            clusters_3d_path = os.path.join(BASE_DIR, 'static', 'keyword_3d_clusters.png')
                            plt.savefig(clusters_3d_path, bbox_inches='tight')
                            plt.close()
                            
                            results['clusters3d_path'] = '/static/keyword_3d_clusters.png'
        except Exception as clusters_3d_error:
            print(f"키워드 군집 3D 시각화 생성 오류: {clusters_3d_error}")
            results['clusters3d_path'] = ''
        
        return jsonify(results)
        
    except Exception as e:
        print(f"분석 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_csv', methods=['POST'])
def download_csv():
    try:
        # 클라이언트에서 보낸 데이터 가져오기
        data = request.json
        
        # 데이터가 없으면 에러 반환
        if not data:
            return jsonify({"error": "데이터가 없습니다."}), 400
        
        # 데이터를 DataFrame으로 변환
        df = pd.DataFrame(data)
        
        # DataFrame을 CSV로 변환
        csv_data = io.StringIO()
        df.to_csv(csv_data, index=False, encoding='utf-8-sig')
        
        # CSV 파일로 반환
        mem = io.BytesIO()
        mem.write(csv_data.getvalue().encode('utf-8-sig'))
        mem.seek(0)
        
        return send_file(
            mem,
            mimetype='text/csv; charset=utf-8',
            as_attachment=True,
            download_name='텍스트_분석_결과.csv'
        )
    except Exception as e:
        app.logger.error(f"CSV 다운로드 오류: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    try:
        # 클라이언트에서 보낸 HTML 내용
        html_content = request.form.get('html_content')
        
        if not html_content:
            return jsonify({"error": "HTML 내용이 없습니다."}), 400
        
        # 기본 HTML 템플릿에 분석 콘텐츠 추가
        full_html = f'''<!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>텍스트 마이닝 분석 결과</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .result-section {{
                    margin-top: 2rem;
                    padding: 1rem;
                    border: 1px solid #dee2e6;
                    border-radius: 0.25rem;
                }}
                body {{
                    font-family: 'Malgun Gothic', sans-serif;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <div class="container mt-5">
                <h1 class="mb-4">텍스트 마이닝 분석 결과</h1>
                {html_content}
            </div>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>'''
        
        # 이미지 URL을 상대 경로로 수정 (static/xxx.png 형식)
        modified_html = full_html.replace('src="/static/', 'src="static/')
        
        # ZIP 파일 생성 (HTML + 이미지)
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            # HTML 파일 추가
            zf.writestr('텍스트_분석_결과.html', modified_html)
            
            # 이미지 파일 추가
            static_dir = os.path.join(BASE_DIR, 'static')
            # HTML에서 사용된 이미지 파일 추출 - 모든 /static/ 경로 찾기
            image_paths = re.findall(r'src="/static/([^"]+)"', html_content)
            
            for img_path in image_paths:
                file_path = os.path.join(static_dir, img_path)
                if os.path.exists(file_path):
                    zf.write(file_path, f'static/{img_path}')
            
            # 기본 이미지들도 추가 (혹시 빠진 이미지가 있을 경우)
            for img_name in ['wordcloud.png', 'network.png', 'heatmap.png',
                             'wordcloud_positive.png', 'wordcloud_negative.png',
                             'topic_shares.png', 'centrality_analysis.png',
                             'clustering_analysis.png', 'keyword_influence_bubble.png',
                             'keyword_3d_clusters.png']:
                img_path = os.path.join(static_dir, img_name)
                if os.path.exists(img_path) and img_name not in image_paths:
                    zf.write(img_path, f'static/{img_name}')
        
        # 메모리 파일 포인터를 처음으로 되돌림
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='텍스트_분석_결과.zip'
        )
    
    except Exception as e:
        app.logger.error(f"ZIP 다운로드 오류: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=False, use_reloader=False)