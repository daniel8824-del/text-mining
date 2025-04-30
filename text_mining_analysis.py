import pandas as pd  #데이터 처리 라이브러리
import numpy as np  #수치 연산 라이브러리
import re  #정규 표현식 라이브러리
import nltk  #자연어 처리 라이브러리 
import warnings  #경고 메시지 무시 라이브러리
import os  #운영체제 관련 라이브러리   
from nltk.tokenize import word_tokenize  #토큰화 라이브러리
from nltk.corpus import stopwords  #불용어 라이브러리
from nltk.stem import WordNetLemmatizer  #표제어 추출 라이브러리
from collections import Counter, defaultdict  #카운터 라이브러리
from sklearn.feature_extraction.text import TfidfVectorizer  #TF-IDF 벡터화 라이브러리
from sklearn.decomposition import LatentDirichletAllocation  #LDA 모델 라이브러리
from gensim import corpora, models  #토픽 모델링 라이브러리
import matplotlib.pyplot as plt  #시각화 라이브러리
import seaborn as sns  #시각화 라이브러리
import networkx as nx  #네트워크 그래프 라이브러리
from wordcloud import WordCloud  #워드 클라우드 라이브러리
from text_mining_knu import KnuTextMining  # 감성사전 로드 

# 한국어 처리 라이브러리
from konlpy.tag import Okt, Hannanum, Mecab, Kkma  # 한국어 형태소 분석기
from soynlp.word import WordExtractor  # 한국어 단어 추출기
from soynlp.tokenizer import LTokenizer  # 한국어 토크나이저
warnings.filterwarnings('ignore')

# NLTK 필요 데이터 다운로드
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# 텍스트 전처리
class TextMiningAnalysis:
    def __init__(self, file_path=None, text_column=None, language='korean'):
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        
        # 한국어 처리 설정
        self.stop_words = self._load_stopwords()
        self.okt = Okt()
        self.tokenizer = self.okt.morphs
        
        # 감정 사전 로드
        self.knu_dict = self._load_sentiment_dict()
        self.pos_words = {word for word, score in self.knu_dict.items() if score > 0}
        self.neg_words = {word for word, score in self.knu_dict.items() if score < 0}

        # 데이터 폴더 지정
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)

        # 텍스트 데이터 로드
        self.corpus = []
        if file_path:
            full_path = os.path.join(self.data_dir, os.path.basename(file_path))
            # 파일 타입에 따라 처리
            if file_path.endswith('.csv'):
                df = pd.read_csv(full_path)
                self.df = df
                self.corpus = df[text_column].dropna().tolist()
            else:
                with open(full_path, 'r', encoding='utf-8') as file:
                    self.raw_text = file.read()
                self.corpus = [self.raw_text]

        # 분석 결과 저장 변수
        self.processed_corpus = []
        self.tokenized_corpus = []
        self.pos_tagged_corpus = []
        self.tf_idf_matrix = None
        self.tf_idf_feature_names = None
        self.lda_model = None
        self.topics = None
    
    def _load_stopwords(self):
        # 불용어 파일 경로
        try:
            # 기본 상대 경로 시도
            stopwords_path = 'korean_stopwords.txt'
            
            # 상대 경로로 파일이 없을 경우 절대 경로 시도
            if not os.path.exists(stopwords_path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                stopwords_path = os.path.join(base_dir, 'korean_stopwords.txt')
                
            if os.path.exists(stopwords_path):
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    return set(f.read().splitlines())
            else:
                print(f"불용어 파일을 찾을 수 없습니다: {stopwords_path}")
                # 불용어 파일을 찾을 수 없는 경우 기본 불용어 제공
                return set(['은', '는', '이', '가', '을', '를', '와', '과', '의', '에', '에서', '로', '으로'])
        except Exception as e:
            print(f"불용어 파일 로드 오류: {e}")
            # 불용어 파일 로드 실패 시 기본 불용어 제공
            return set(['은', '는', '이', '가', '을', '를', '와', '과', '의', '에', '에서', '로', '으로'])
            
    def _load_sentiment_dict(self):
        # 감성사전 파일 경로
        try:
            sentiment_path = 'knu_sentiment_lexicon.csv'
            
            # 상대 경로로 파일이 없을 경우 절대 경로 시도
            if not os.path.exists(sentiment_path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                sentiment_path = os.path.join(base_dir, 'knu_sentiment_lexicon.csv')
                
            sentiment_dict = {}
            
            if os.path.exists(sentiment_path):
                with open(sentiment_path, 'r', encoding='utf-8') as f:
                    next(f)  # 헤더 스킵
                    for line in f:
                        if ',' in line:
                            try:
                                word, score = line.strip().split(',')
                                if word and score:
                                    sentiment_dict[word] = float(score)
                            except ValueError:
                                continue  # 라인 파싱 오류 무시
                return sentiment_dict
            else:
                print(f"감성사전 파일을 찾을 수 없습니다: {sentiment_path}")
                # KNU 감성사전 대신 기본 감성사전 제공
                return {'좋다': 1.0, '나쁘다': -1.0, '행복': 1.0, '슬픔': -1.0}
        except Exception as e:
            print(f"감성사전 로드 오류: {e}")
            return {'좋다': 1.0, '나쁘다': -1.0, '행복': 1.0, '슬픔': -1.0}
    
    def preprocess_text(self, min_word_length=2, pos_filter=None, custom_stopwords=None):
        """텍스트 전처리: 특수문자 제거, 토큰화, 불용어 제거, 짧은 단어 제거"""
        self.processed_corpus = []
        self.tokenized_corpus = []
        self.pos_tagged_corpus = []
        
        # 한국어 처리를 위한 okt 초기화
        okt = Okt()
        
        # 불용어 설정
        stop_words = set(stopwords.words('english'))
        
        # 기본 불용어 추가 (사람 이름, 채널명 등) - 내부에서만 처리
        default_stopwords = ['shorts', 'short', '쇼츠', 'viral', 'youtube', '유튜브', '채널', 
                            '구독', '좋아요', 'sub', 'subscribe', 'like', 'com', 'kr',
                            'tv', '티비', '방송', '실시간', '라이브', 'live', 'tv']
        stop_words.update(default_stopwords)
        
        # 웹에서 전달받은 사용자 정의 불용어 추가
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        
        # 문서별 처리
        for doc in self.corpus:
            # 한국어 정규화
            doc = self._normalize_korean(doc)
            
            # 특수문자 제거
            doc_clean = re.sub(r'[^\w\s]', ' ', doc)
            
            # 품사 태깅 및 토큰화
            if pos_filter:
                # 지정된 품사만 추출
                pos_tagged = okt.pos(doc_clean)
                self.pos_tagged_corpus.append(pos_tagged)
                
                tokens = [word for word, pos in pos_tagged 
                          if pos in pos_filter 
                          and len(word) >= min_word_length 
                          and word not in stop_words]
            else:
                # 일반 토큰화
                tokens = okt.morphs(doc_clean)
                tokens = [token for token in tokens 
                          if len(token) >= min_word_length 
                          and token not in stop_words]
            
            # 결과 저장
            self.tokenized_corpus.append(tokens)
            self.processed_corpus.append(' '.join(tokens))
        
        return self.processed_corpus
    
    def _normalize_korean(self, text):
         # 한국어 텍스트 정규화
        # 반복되는 한글 자음/모음 정규화 ('ㅋㅋㅋㅋ' -> 'ㅋㅋ')
        # 반복되는 단어 정규화 ('완전완전' -> '완전')
        # 반복되는 한글 자음/모음 정규화 (3개 이상 반복 -> 2개로)
        pattern = re.compile(r'([ㄱ-ㅎㅏ-ㅣ])\1{2,}')
        while pattern.search(text):
            text = pattern.sub(r'\1\1', text)
        
        pattern = re.compile(r'([가-힣])\1{2,}')
        while pattern.search(text):
            text = pattern.sub(r'\1\1', text)
            
        return text
        
    def extract_keywords(self, top_n=20):
        # 키워드 추출
        all_words = []
        for tokens in self.tokenized_corpus: all_words.extend(tokens)
        word_freq = Counter(all_words)
        return dict(word_freq.most_common(top_n))
    
    def perform_tf_idf_analysis(self, max_features=1000):
        # TF-IDF 분석
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        self.tf_idf_matrix = tfidf_vectorizer.fit_transform(self.processed_corpus)
        self.tf_idf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    def get_top_tf_idf_keywords(self, top_n=20):
        # TF-IDF 기반 상위 키워드 추출
        if self.tf_idf_matrix is None:
            self.perform_tf_idf_analysis()
            
        sorted_indices = np.argsort(self.tf_idf_matrix.toarray(), axis=1)
        top_keywords = []
        
        for idx, indices in enumerate(sorted_indices):
            top_indices = indices[-top_n:][::-1]
            keywords = [(self.tf_idf_feature_names[i], self.tf_idf_matrix[idx, i]) 
                        for i in top_indices]
            top_keywords.append(keywords)
            
        return top_keywords
    
    def topic_modeling(self, num_topics=5, num_words=10):
        # LDA 토픽 모델링
        if self.tf_idf_matrix is None:
            self.perform_tf_idf_analysis()
            
        lda = LatentDirichletAllocation(n_components=num_topics, 
                                        random_state=42)
        self.lda_model = lda.fit(self.tf_idf_matrix)
        
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-num_words:][::-1]
            top_words = [(self.tf_idf_feature_names[i], topic[i]) for i in top_words_idx]
            topics.append(top_words)
            
        self.topics = topics
        return topics
    
    def create_word_cloud(self, width=800, height=400, font_path=None):
        # 워드 클라우드 생성
        all_words = ' '.join(self.processed_corpus)
        
        wordcloud = WordCloud(
            font_path=font_path,
            width=width, 
            height=height, 
            background_color='white',
            max_words=200
        ).generate(all_words)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        
        return wordcloud
    
    def keyword_network_analysis(self, threshold=2, top_n=50):
        """키워드 네트워크 분석
        
        Args:
            threshold: 동시 출현 빈도 임계값
            top_n: 분석에 사용할 상위 키워드 수
            
        Returns:
            nx.Graph: 키워드 네트워크 그래프
        """
        if not self.tokenized_corpus:
            print("텍스트를 먼저 전처리해야 합니다.")
            return None
        
        # 키워드 빈도 계산
        all_words = [word for doc in self.tokenized_corpus for word in doc]
        word_counts = Counter(all_words)
        
        # 상위 N개 키워드 선택
        top_words = [word for word, count in word_counts.most_common(top_n)]
        
        # 동시 출현 빈도 계산
        co_occurrence = defaultdict(int)
        for doc in self.tokenized_corpus:
            # 문서에 포함된 상위 키워드만 고려
            words_in_doc = [word for word in doc if word in top_words]
            
            # 동시 출현 쌍 생성
            for i, word1 in enumerate(words_in_doc):
                for word2 in words_in_doc[i+1:]:
                    if word1 != word2:
                        pair = tuple(sorted([word1, word2]))
                        co_occurrence[pair] += 1
        
        # 네트워크 생성
        G = nx.Graph()
        
        # 노드 추가
        for word in top_words:
            G.add_node(word, size=word_counts[word])
        
        # 엣지 추가
        for (word1, word2), weight in co_occurrence.items():
            if weight >= threshold:
                G.add_edge(word1, word2, weight=weight)
        
        # 가장 큰 연결 요소만 추출
        if not nx.is_connected(G):
            print(f"네트워크가 연결되어 있지 않습니다. 전체 컴포넌트 수: {nx.number_connected_components(G)}")
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            print(f"가장 큰 연결 요소만 사용합니다. 노드 수: {len(G.nodes)}")
        
        return G
    
    def plot_network(self, graph, figsize=(12, 12)):
        # 키워드 네트워크 시각화
        plt.figure(figsize=figsize)
        
        # 한글 폰트 설정 (시스템에 적합한 폰트 선택)
        try:
            if os.path.exists('C:/Windows/Fonts/malgun.ttf'):  # Windows
                plt.rcParams['font.family'] = 'Malgun Gothic'
            else:
                # Linux/Mac에서는 운영체제에 설치된 기본 폰트 사용
                plt.rcParams['font.family'] = 'sans-serif'
        except Exception:
            # 폰트 설정 오류 발생 시 기본 폰트 사용
            pass
            
        plt.rcParams['axes.unicode_minus'] = False
        
        # 노드 크기와 엣지 두께 설정
        node_sizes = [graph.nodes[node]['size'] * 30 for node in graph.nodes()]
        edge_weights = [graph.edges[edge]['weight'] for edge in graph.edges()]
        
        # 레이아웃 설정
        pos = nx.spring_layout(graph, k=0.3)
        
        # 그래프 그리기
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.8)
        nx.draw_networkx_edges(graph, pos, width=edge_weights, alpha=0.5)
        nx.draw_networkx_labels(graph, pos, font_size=12, font_family='Malgun Gothic') #sans-serif->Malgun Gothic
        
        plt.axis('off')
        plt.tight_layout()
    
    def sentiment_analysis(self):
        # 감정 분석
        results = []
        
        for i, doc in enumerate(self.corpus):
            tokens = self.tokenizer(doc)
            sentiment_scores = [self.knu_dict.get(word, 0) for word in tokens]
            pos_count = sum(1 for score in sentiment_scores if score > 0)
            neg_count = sum(1 for score in sentiment_scores if score < 0)
            total_score = sum(sentiment_scores)
            polarity = 0
            if pos_count + neg_count > 0:
                polarity = total_score / (pos_count + neg_count)
            
            # 감정 레이블
            sentiment = 'Positive' if polarity > 0.1 else 'Negative' if polarity < -0.1 else 'Neutral'
            
            # 결과 저장
            results.append({
                'Document': i,
                'Polarity': polarity,
                'Sentiment': sentiment,
                'Positive_Words': pos_count,
                'Negative_Words': neg_count
            })
        
        return pd.DataFrame(results)
    
    def plot_sentiment_distribution(self, sentiment_df):
        # 감정 분포 시각화
        plt.figure(figsize=(12, 6))
        
        # 감정 분포
        plt.subplot(1, 2, 1)
        sentiment_counts = sentiment_df['Sentiment'].value_counts()
        plt.pie(sentiment_counts, labels=sentiment_counts.index, 
                autopct='%1.1f%%', colors=['green', 'grey', 'red'])
        plt.title('Sentiment Distribution')
        
        # 극성 분포
        plt.subplot(1, 2, 2)
        sns.histplot(sentiment_df['Polarity'], bins=30, kde=True)
        plt.title('Polarity Distribution')
        plt.xlabel('Polarity')
        
        plt.tight_layout()

    @classmethod
    def from_csv(cls, csv_file, text_column, language='korean'):
        # CSV 파일 로드
        return cls(file_path=csv_file, text_column=text_column, language=language)
        
    @classmethod
    def from_excel(cls, excel_file, text_column, sheet_name=0, language='korean', **kwargs):
        # 데이터 폴더 경로 설정
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)
        
        # 파일 경로 설정
        full_path = os.path.join(data_dir, os.path.basename(excel_file))
        
        # Excel 파일 로드 및 CSV로 변환
        df = pd.read_excel(full_path, sheet_name=sheet_name, **kwargs)
        
        # CSV 파일로 저장
        csv_file = os.path.splitext(full_path)[0] + '.csv'
        df.to_csv(csv_file, index=False)
        
        # TextMiningAnalysis 객체 생성 및 반환
        return cls(file_path=csv_file, text_column=text_column, language=language)

# 예제 사용법
if __name__ == "__main__":
    # 1. 텍스트 분석기 초기화
    analyzer = TextMiningAnalysis(file_path='data/Youtube.csv', text_column='제목')
    
    # 2. 품사 태깅 설정
    pos_options = {
        '1': 'Noun', '2': 'Verb', '3': 'Adjective',
        '4': 'Adverb', '5': 'Determiner'
    }
    
    print("\n=== 텍스트 분석 옵션 ===")
    print("1: 명사\n2: 동사\n3: 형용사\n4: 부사\n5: 관형사")
    
    user_input = input("\n분석할 품사를 선택하세요(쉼표로 구분, 예: 1,2): ").strip()
    
    if user_input:
        selected_pos = [pos_options[pos.strip()] 
                       for pos in user_input.split(',') 
                       if pos.strip() in pos_options]
        analyzer.preprocess_text(pos_filter=selected_pos)
    else:
        analyzer.preprocess_text()
    
    # 3. 분석 결과 출력
    print("\n=== 분석 결과 ===")
    
    # 4.1 키워드 추출 (상위 20개)
    print("1. 주요 키워드(빈도수 상위 20개):")
    keywords = analyzer.extract_keywords(top_n=20)
    for word, freq in keywords.items():
        print(f"- {word}: {freq}회")
    
    # 4.2 TF-IDF 키워드 추출
    print("\n2. 주요 키워드(TF-IDF 상위 20개):")
    analyzer.perform_tf_idf_analysis()
    tfidf_keywords = analyzer.get_top_tf_idf_keywords(top_n=20)
    
    # 전체 문서의 TF-IDF 키워드 통합
    all_tfidf_keywords = {}
    for doc_keywords in tfidf_keywords:
        for word, score in doc_keywords:
            if word in all_tfidf_keywords:
                all_tfidf_keywords[word] = max(all_tfidf_keywords[word], score)
            else:
                all_tfidf_keywords[word] = score
    
    # 상위 20개 키워드 출력
    sorted_keywords = sorted(all_tfidf_keywords.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_keywords[:20]:
        print(f"- {word}: {score:.4f}")
    
    # 4.3 토픽 모델링
    print("\n3. 토픽 모델링 결과:")
    topics = analyzer.topic_modeling(num_topics=5, num_words=10)
    
    # 토픽 자동 제목 생성
    topic_titles = []
    
    # 키워드 가중치 반영 방식으로 토픽 이름 생성 (선택 없이 바로 적용)
    print("\n토픽 이름을 키워드 가중치를 반영하여 생성합니다...")
    for topic_keywords in topics:
        # 가중치 합계 계산
        total_weight = sum([weight for _, weight in topic_keywords[:3]])
        
        # 가중치 비율로 표현
        title_parts = []
        for word, weight in topic_keywords[:3]:
            percentage = int(100 * weight / total_weight)
            title_parts.append(f"{word}({percentage}%)")
            
        topic_titles.append(' + '.join(title_parts))
    
    # 토픽 결과 출력
    for i, (topic, title) in enumerate(zip(topics, topic_titles)):
        print(f"\n토픽 {i+1}: {title}")
        for word, score in topic:
            print(f"- {word}: {score:.4f}")
    
    # 4.4 키워드 네트워크 분석
    print("\n4. 키워드 네트워크 분석:")
    network = analyzer.keyword_network_analysis(threshold=2, top_n=50)
    print(f"- 총 {len(network.nodes)}개의 노드")
    print(f"- 총 {len(network.edges)}개의 연결")
    
    # 4.5 감정 분석 - 불필요한 부분 제거하고 바로 긍/부정 분석으로 변경
    # 4.6 긍/부정 분석
    print("\n5. 긍/부정 분석:")
    sentiment = analyzer.sentiment_analysis()
    positive_docs = sentiment[sentiment['Sentiment'] == 'Positive']
    negative_docs = sentiment[sentiment['Sentiment'] == 'Negative']
    neutral_docs = sentiment[sentiment['Sentiment'] == 'Neutral']
    
    print(f"- 긍정 문서 수: {len(positive_docs)} ({len(positive_docs)/len(sentiment)*100:.1f}%)")
    print(f"- 부정 문서 수: {len(negative_docs)} ({len(negative_docs)/len(sentiment)*100:.1f}%)")
    print(f"- 중립 문서 수: {len(neutral_docs)} ({len(neutral_docs)/len(sentiment)*100:.1f}%)")
    
    # 긍정/부정 단어 빈도수 계산
    pos_word_counts = {}
    neg_word_counts = {}
    
    for doc in analyzer.corpus:
        tokens = analyzer.tokenizer(doc)
        for token in tokens:
            if token in analyzer.pos_words:
                pos_word_counts[token] = pos_word_counts.get(token, 0) + 1
            elif token in analyzer.neg_words:
                neg_word_counts[token] = neg_word_counts.get(token, 0) + 1
    
    # 6. 상위 긍정 단어 출력
    print("\n6. 긍정 단어 상위 빈도수 5개:")
    top_pos_words = sorted(pos_word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for word, count in top_pos_words:
        score = analyzer.knu_dict.get(word, 0)
        print(f"- {word}: {count}회 (감성 점수: {score:.1f})")
    
    # 7. 상위 부정 단어 출력
    print("\n7. 부정 단어 상위 빈도수 5개:")
    top_neg_words = sorted(neg_word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for word, count in top_neg_words:
        score = analyzer.knu_dict.get(word, 0)
        print(f"- {word}: {count}회 (감성 점수: {score:.1f})")
    
    # 8. 시각화
    print("\n8. 시각화 결과 저장 중...")
    os.makedirs('visualization_results', exist_ok=True)
    
    # 8.1 전체 워드 클라우드
    print("- 전체 키워드 워드 클라우드 생성 중...")
    wordcloud = analyzer.create_word_cloud(font_path='C:/Windows/Fonts/malgun.ttf')
    wordcloud.to_file('visualization_results/wordcloud_all.png')
    
    # 8.2 키워드 네트워크 (한글 폰트 적용)
    print("- 키워드 네트워크 생성 중...")
    plt.figure(figsize=(15, 15))
    # 한글 폰트 설정은 plot_network 함수 내부에 포함되어 있음
    analyzer.plot_network(network)
    plt.savefig('visualization_results/network.png')
    plt.close()
    
    # 8.3 긍정 단어 워드 클라우드
    print("- 긍정 단어 워드 클라우드 생성 중...")
    pos_words_text = ' '.join([f"{word} " * count for word, count in pos_word_counts.items()])
    if pos_words_text.strip():
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
        plt.title('긍정 단어 워드 클라우드', fontsize=15, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('visualization_results/wordcloud_positive.png')
        plt.close()
    
    # 8.4 부정 단어 워드 클라우드
    print("- 부정 단어 워드 클라우드 생성 중...")
    neg_words_text = ' '.join([f"{word} " * count for word, count in neg_word_counts.items()])
    if neg_words_text.strip():
        neg_wordcloud = WordCloud(
            font_path='C:/Windows/Fonts/malgun.ttf',
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='OrRd'
        ).generate(neg_words_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(neg_wordcloud, interpolation='bilinear')
        plt.title('부정 단어 워드 클라우드', fontsize=15, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('visualization_results/wordcloud_negative.png')
        plt.close()
    
    # 8.5 토픽 점유율 그래프
    print("- 토픽 점유율 그래프 생성 중...")
    plt.figure(figsize=(12, 6))
    # 토픽 분포 계산
    topic_distribution = analyzer.lda_model.transform(analyzer.tf_idf_matrix)
    topic_shares = topic_distribution.mean(axis=0)
    # 토픽별 색상 설정
    colors = plt.cm.tab10(np.arange(len(topic_shares)))
    
    # 토픽 이름 생성
    topic_names = [f"토픽 {i+1}: {title}" for i, title in enumerate(topic_titles)]
    
    # 파이 차트 생성
    plt.pie(topic_shares, labels=topic_names, autopct='%1.1f%%', 
            colors=colors, startangle=90, shadow=True)
    plt.title('토픽 점유율', fontsize=16, pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('visualization_results/topic_shares.png')
    plt.close()
    
    # 8.6 중심성 분석
    print("- 키워드 중심성 분석 생성 중...")
    plt.figure(figsize=(12, 8))
    
    # 중심성 계산
    degree_centrality = nx.degree_centrality(network)
    betweenness_centrality = nx.betweenness_centrality(network, k=10)
    
    # 연결되지 않은 그래프에서도 작동하도록 예외 처리
    try:
        eigenvector_centrality = nx.eigenvector_centrality_numpy(network)
    except nx.NetworkXError:
        # 대체 방법: 가장 큰 연결 요소에서만 고유벡터 중심성 계산
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
    except Exception as e:
        print(f"고유벡터 중심성 계산 중 오류 발생: {e}")
        # 고유벡터 중심성 대신 연결 중심성으로 대체
        eigenvector_centrality = degree_centrality
    
    # 상위 10개 노드만 선택
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_nodes = [item[0] for item in top_degree]
    
    # 각 중심성 값 정규화
    degree_vals = [degree_centrality[node] for node in top_nodes]
    betweenness_vals = [betweenness_centrality[node] for node in top_nodes]
    eigenvector_vals = [eigenvector_centrality[node] for node in top_nodes]
    
    # 그래프 그리기
    x = range(len(top_nodes))
    width = 0.25  # 막대 너비
    
    plt.bar([i - width for i in x], degree_vals, width, label='연결 중심성', color='skyblue')
    plt.bar(x, betweenness_vals, width, label='매개 중심성', color='lightgreen')
    plt.bar([i + width for i in x], eigenvector_vals, width, label='고유벡터 중심성', color='salmon')
    
    plt.xlabel('키워드', fontsize=12)
    plt.ylabel('중심성 값', fontsize=12)
    plt.title('키워드 중심성 분석 (상위 10개)', fontsize=16, pad=20)
    plt.xticks(x, top_nodes, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualization_results/centrality_analysis.png')
    plt.close()
    
    # 8.7 클러스터링 시각화
    print("- 키워드 클러스터링 시각화 생성 중...")
    plt.figure(figsize=(15, 15))
    
    # 커뮤니티 탐지
    communities = list(nx.algorithms.community.greedy_modularity_communities(network))
    
    # 노드별 커뮤니티 할당
    node_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_community[node] = i
    
    # 노드 크기와 엣지 두께 설정
    try:
        node_sizes = [network.nodes[node]['size'] * 30 for node in network.nodes()]
        edge_weights = [network.edges[edge]['weight'] for edge in network.edges()]
    
        # 각 커뮤니티별 색상 설정
        community_colors = [plt.cm.tab20(i) for i in range(len(communities))]
        node_colors = [community_colors[node_community.get(node, 0)] for node in network.nodes()]
    
        # 레이아웃 설정
        pos = nx.spring_layout(network, k=0.3, seed=42)
    
        # 그래프 그리기
        nx.draw_networkx_nodes(network, pos, node_size=node_sizes, 
                            node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(network, pos, width=edge_weights, alpha=0.3)
        nx.draw_networkx_labels(network, pos, font_size=10, font_family='Malgun Gothic') #sans-serif->Malgun Gothic 
    
        # 커뮤니티 라벨 추가
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
                        horizontalalignment='center', fontsize=12)
    
        plt.axis('off')
        plt.title('키워드 클러스터 분석', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('visualization_results/clustering_analysis.png')
        plt.close()
    except Exception as e:
        print(f"클러스터링 시각화 생성 중 오류 발생: {e}")
    
    # 8.8 키워드 상관관계 히트맵
    print("- 키워드 상관관계 히트맵 생성 중...")
    try:
        # 상위 15개 키워드 선택
        top_keywords = list(keywords.keys())[:15]
        
        # 키워드 상관관계 계산
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
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', 
                    xticklabels=top_keywords, yticklabels=top_keywords)
        plt.title('키워드 상관관계 히트맵', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('visualization_results/keyword_correlation_heatmap.png')
        plt.close()
    except Exception as e:
        print(f"히트맵 생성 중 오류 발생: {e}")
    
    # 8.9 키워드 영향력 버블 차트
    print("- 키워드 영향력 버블 차트 생성 중...")
    try:
        # 중심성 분석 결과 사용
        plt.figure(figsize=(12, 8))
        
        # 상위 20개 노드 선택
        top_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
        nodes = [item[0] for item in top_centrality]
        
        # 각 노드의 크기, 색상 결정 (eigenvector_centrality는 이미 계산된 상태)
        sizes = [network.nodes[node]['size'] * 100 for node in nodes]
        colors = [eigenvector_centrality[node] for node in nodes]
        
        # 버블 차트 생성
        x = [degree_centrality[node] for node in nodes]  # 연결 중심성
        y = [betweenness_centrality[node] for node in nodes]  # 매개 중심성
        
        scatter = plt.scatter(x, y, s=sizes, c=colors, alpha=0.7, cmap='viridis', 
                             edgecolors='gray', linewidths=1)
        
        # 각 노드에 라벨 추가
        for i, node in enumerate(nodes):
            plt.annotate(node, (x[i], y[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, alpha=0.8)
        
        plt.colorbar(scatter, label='고유벡터 중심성')
        plt.xlabel('연결 중심성', fontsize=12)
        plt.ylabel('매개 중심성', fontsize=12)
        plt.title('키워드 영향력 버블 차트', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('visualization_results/keyword_influence_bubble.png')
        plt.close()
    except Exception as e:
        print(f"버블 차트 생성 중 오류 발생: {e}")
        
    # 8.10 키워드 군집 분석 3D 시각화 (선택 사항)
    try:
        from sklearn.manifold import TSNE
        from mpl_toolkits.mplot3d import Axes3D
        
        print("- 키워드 군집 3D 시각화 생성 중...")
        
        # 상위 100개 키워드의 임베딩 벡터 생성 (TF-IDF 행렬 사용)
        top_words = list(keywords.keys())[:100]
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
            
            # 커뮤니티 색상 매핑
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
                if word in word_to_community and i < len(word_indices):
                    colors.append(word_to_community[word])
                    sizes.append(keywords.get(word, 1) * 20)
                    visible_words.append(word)
                    visible_coords.append(word_vectors_3d[i])
            
            if visible_coords:
                visible_coords = np.array(visible_coords)
                scatter = ax.scatter(
                    visible_coords[:, 0], visible_coords[:, 1], visible_coords[:, 2],
                    c=colors, s=sizes, alpha=0.7, cmap='tab20'
                )
                
                # 주요 키워드에만 라벨 추가
                for i, word in enumerate(visible_words):
                    if keywords.get(word, 0) > np.percentile([keywords.get(w, 0) for w in visible_words], 70):
                        ax.text(visible_coords[i, 0], visible_coords[i, 1], visible_coords[i, 2], 
                               word, fontsize=9)
                
                ax.set_title('키워드 3D 군집 시각화', fontsize=16, pad=20)
                ax.set_xlabel('Dimension 1')
                ax.set_ylabel('Dimension 2')
                ax.set_zlabel('Dimension 3')
                plt.tight_layout()
                plt.savefig('visualization_results/keyword_3d_clusters.png')
                plt.close()
    except Exception as e:
        print(f"3D 시각화 생성 중 오류 발생: {e}")
    
    print("시각화 결과가 'visualization_results' 폴더에 저장되었습니다.") 