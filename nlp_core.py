# -*- coding: utf-8 -*-

import json
import nltk
import joblib
import random
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Bloco de verificação e download dos recursos do NLTK (mantido como está, é uma ótima abordagem)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Recurso 'punkt' não encontrado. Baixando agora...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Recurso 'stopwords' não encontrado. Baixando agora...")
    nltk.download('stopwords')

try:
    nltk.data.find('stemmers/rslp')
except LookupError:
    print("Recurso 'rslp' (stemmer) não encontrado. Baixando agora...")
    nltk.download('rslp')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Recurso 'punkt_tab' não encontrado. Baixando agora...")
    nltk.download('punkt_tab')


class NLPModel:
    """
    Classe refatorada para usar um Pipeline do Scikit-learn, unificando
    o vetorizador e o classificador em um único objeto.
    """

    def __init__(self, data_file='database.json', model_file='modelo_pipeline.pkl'):
        """
        Inicializa a classe, definindo os caminhos e carregando os dados.
        """
        self.data_file = data_file
        self.model_file = model_file
        self.stopwords = nltk.corpus.stopwords.words('portuguese')
        self.stemmer = nltk.stem.RSLPStemmer()
        self.model = None
        self._load_data()

    def _load_data(self):
        """ Carrega as intenções do arquivo JSON. """
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.intents = json.load(f)

    def _preprocess(self, text):
        """
        Realiza o pré-processamento de um texto: tokenização, stemming e remoção de stopwords.
        """
        text = text.lower()
        tokens = nltk.word_tokenize(text, language='portuguese')
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stopwords and word.isalpha()]
        return " ".join(stemmed_tokens)

    def train(self):
        """
        Prepara os dados, treina o pipeline (vetorizador + classificador) e o salva.
        """
        print("Iniciando o treinamento do modelo...")

        # Preparar dados de treino
        documents = []
        tags = []
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                documents.append(self._preprocess(pattern))
                tags.append(intent['tag'])

        # Divisão de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(documents, tags, test_size=0.2, random_state=42, stratify=tags)
        
        print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

        # Criação do Pipeline: une o vetorizador e o classificador
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
            ('classifier', LogisticRegression(random_state=42, C=1.0, solver='lbfgs', max_iter=500))
        ])

        # Treinamento do pipeline inteiro
        self.model.fit(X_train, y_train)

        # Avaliação mais detalhada do modelo
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAcurácia do modelo nos dados de teste: {accuracy:.2%}\n")
        print("Relatório de Classificação Detalhado:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Salvar o pipeline inteiro em um único arquivo
        joblib.dump(self.model, self.model_file)

        print(f"Modelo (pipeline) salvo em '{self.model_file}'")
        print("Treinamento concluído com sucesso!")

    def load_model(self):
        """
        Carrega o pipeline completo a partir de um único arquivo.
        """
        try:
            self.model = joblib.load(self.model_file)
            print("Pipeline do modelo carregado com sucesso.")
        except FileNotFoundError:
            print(f"Erro: Arquivo do modelo '{self.model_file}' não encontrado. Execute o treinamento primeiro.")
            raise

    def predict_intent(self, text):
        """
        Prevê a intenção de um dado texto usando o pipeline carregado.
        Se o texto parece conter múltiplas intenções, tentará identificar e
        responder a cada uma delas.
        """
        if self.model is None:
            self.load_model()
        
        # O pré-processamento precisa ser feito antes de passar para o pipeline
        processed_text = self._preprocess(text)
        
        # Análise de palavras-chave por categoria
        keywords = {
            'preço': ['preço', 'preco', 'valor', 'custa', 'custo', 'quanto'],
            'cardápio': ['cardapio', 'cardápio', 'menu', 'opções', 'opcoes', 'lanches', 'hamburgueres', 'burgers', 'combo', 'lanche'],
            'bebida': ['bebida', 'refrigerante', 'suco', 'água', 'agua', 'refri', 'cooler', 'cerveja', 'milkshake', 'shake'],
            'sobremesa': ['sobremesa', 'doce', 'sorvete', 'sobremesas', 'açucar', 'açúcar', 'chocolate'],
            'entrega': ['entrega', 'demora', 'tempo', 'frete', 'delivery', 'quando chega']
        }
        
        # Identificar quais categorias estão presentes na mensagem
        categories_found = {}
        for category, words in keywords.items():
            if any(word in text.lower() for word in words):
                categories_found[category] = True
        
        # Se encontrar mais de uma categoria, processa como múltiplas intenções
        if len(categories_found) > 1:
            # Primeiro tenta identificar intenções específicas para cada categoria
            responses = []
            
            # Predição principal (geral)
            prediction_proba = self.model.predict_proba([processed_text])
            best_class_index = np.argmax(prediction_proba[0])
            main_intent = self.model.classes_[best_class_index]
            main_confidence = prediction_proba[0][best_class_index]
            
            # Se a intenção principal for clara (alta confiança), usa ela
            if main_confidence > 0.7:
                # Buscar resposta para a intenção principal
                for intent in self.intents['intents']:
                    if intent['tag'] == main_intent:
                        answer = random.choice(intent['responses'])
                        return main_intent, main_confidence, answer
            
            # Caso contrário, tenta responder cada parte separadamente
            intent_responses = []
            
            # Verificar cada categoria e buscar respostas específicas
            if 'preço' in categories_found:
                for intent in self.intents['intents']:
                    if intent['tag'] == 'ver_preco':
                        intent_responses.append(random.choice(intent['responses']))
                        break
            
            if 'cardápio' in categories_found:
                for intent in self.intents['intents']:
                    if intent['tag'] == 'ver_cardapio_geral':
                        intent_responses.append(random.choice(intent['responses']))
                        break
            
            if 'entrega' in categories_found:
                for intent in self.intents['intents']:
                    if intent['tag'] == 'tempo_entrega':
                        intent_responses.append(random.choice(intent['responses']))
                        break
            
            # Se encontrou respostas específicas, combina-as
            if intent_responses:
                combined_answer = " ".join(intent_responses)
                return "multiplas_intencoes", 0.95, combined_answer
            
            # Se não conseguiu responder especificamente, usa a resposta genérica de múltiplas intenções
            for intent in self.intents['intents']:
                if intent['tag'] == "multiplas_intencoes":
                    answer = random.choice(intent['responses'])
                    return "multiplas_intencoes", 0.95, answer
        
        # Caso contrário, segue com a predição normal
        prediction_proba = self.model.predict_proba([processed_text])
        
        # Pega a melhor classe e sua probabilidade
        best_class_index = np.argmax(prediction_proba[0])
        intent_tag = self.model.classes_[best_class_index]
        confidence = prediction_proba[0][best_class_index]
        
        # Selecionar uma resposta
        answer = "Desculpe, não entendi. Pode reformular?"
        for intent in self.intents['intents']:
            if intent['tag'] == intent_tag:
                answer = random.choice(intent['responses'])
                break
        
        return intent_tag, confidence, answer

if __name__ == '__main__':
    # Bloco para executar o treinamento diretamente
    nlp_model = NLPModel()
    nlp_model.train()

