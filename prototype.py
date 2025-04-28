import os
import re
import PyPDF2
import docx2txt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from gensim.models import Word2Vec


class Parser:
    """Парсит строку, извлекает из нее вызовы инструментов."""
    
    @staticmethod
    def parse_tools(text: str):
        """Парсинг вызовов с выделением инструментов.

        Args:
            text: список запросов.

        Returns:
            список из словарей: 
            'tool' - название инструмента
            'request' - запрос, для которого нужно использовать инструмент.
        """
        
        reg = r'\[(.*?)\]\s*(.*)'
        matches = re.findall(reg, text)
        return [{'tool': match[0], 'request': match[1]} for match in matches]


class FindTool:
    """Инструмент [find], реализующий семантический поиск по базе знаний"""
    
    def __init__(self, path: str):
        """Args:
            path: путь к директории.
        
        Parameters
        ----------
        directory_path : str
            путь к директории
        documents : list
            полученные из директории документы
        texts : list
            извлеченный из документов текст
        """
        
        self.directory_path = path
        self.documents = []
        self.texts = []
        self.vectorizer = TfidfVectorizer()
        self.word_vectors = None
        self.word2vec_model = None

    def load_documents(self):
        """Загружает файлы из директории."""
        
        for root, _, files in os.walk(self.directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                text = self.text_extraction(file_path)
                if text:
                    self.documents.append({
                        'path': file_path,
                        'text': text
                    })
                    self.texts.append(text)

    def text_extraction(self, path: str) -> str:
        """Извлекает текст из файлов с различными разрешениями (pdf, docx, csv, xlsx, txt).

        Args:
            path: путь к директории

        Returns:
            текст, полученный из документа.
        """
        
        if path.endswith('.pdf'):
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = '\n'.join([page.extract_text() for page in reader.pages])
        elif path.endswith('.docx'):
            text = docx2txt.process(path)
        elif path.endswith('.csv'):
            df = pd.read_csv(path)
            text = " ".join(df.to_string().split())
        elif path.endswith('.xlsx'):
            df = pd.read_excel(path)
            text = " ".join(df.to_string().split())
        else:
            with open(path, 'r', encoding='utf-8') as f: 
                text = f.read() #обрабатывает текстовые форматы (.txt)
        return text

    def preprocessing(self, text: str) -> list:
        """Токенизация текста: удаление знаков препинания, вспомогательных слов, разбиение"""
        
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return tokens

    def build_tfidf_model(self):
        """Построение модели TF-IDF, выделение ключевых слов"""
        
        processed = [" ".join(self.preprocessing(text)) for text in self.texts]
        self.vectorizer.fit(processed)

    def build_word2vec_model(self):
        """Построение модели Word2Vec для поиска семантически похожих слов"""
        
        sentences = [self.preprocessing(text) for text in self.texts]
        self.word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    def semantic_search(self, word: str):
        """Осуществляет поиск семантически похожих на заданное слов в тексте.

        Args:
            word: искомое слово

        Returns:
            список семантически похожих слов. 
            в случае отсутствия совпадений выводит соответствующее уведомление.
        """

        
        word_tokens = self.preprocessing(word)
        similar_words = []
        
        for token in word_tokens:
            if token in self.word2vec_model.wv:
                similar = self.word2vec_model.wv.most_similar(token, topn=n)
                similar_words.extend([(wrd, score) for wrd, score in similar])
        
        similar_words = list(set(similar_words))
        similar_words.sort(key=lambda x: x[1], reverse=True)

        if len(similar_words) > 0:
            return [similar_words[i][0] for i in range(0, len(similar_words))]
        return "No matches found"


class NeuroResponse: 
    """Обрабатывает запросы нейросети, использует инструменты."""
    
    def __init__(self, path: str):
        """Args:
            path: путь к директории.
        
        Parameters
        ----------
        search : FindTool class object
            аттрибут для семантического поиска.
        parser : Parser class object
            аттрибут для парсинга запросов.
        """
        
        self.search = FindTool(path)
        self.parser = Parser()

    def response_process(self, text: str):
        """Непосредственно обрабатывает запросы.

        Args:
            text: текст запроса

        Returns:
            результат использования инструментов из запросов.
        """
        
        requests = self.parser.parse_tools(text)
        self.search.load_documents()
        self.search.build_word2vec_model()
        result = []

        for req in requests: 
            if req['tool'] == 'find':
                search_results = self.search.semantic_search(req['request'])
                print(search_results)
