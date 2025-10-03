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
        self.raw_data = {}
        self.intents = []
        self.menu = {}
        self._load_data()

    def _load_data(self):
        """ Carrega as intenções do arquivo JSON. """
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.raw_data = data
        self.intents = data.get('intents', [])
        self.menu = data.get('cardapio_detalhado', {})

    def _preprocess(self, text):
        """
        Realiza o pré-processamento de um texto: tokenização, stemming e remoção de stopwords.
        """
        text = text.lower()
        tokens = nltk.word_tokenize(text, language='portuguese')
        stemmed_tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stopwords and word.isalpha()]
        return " ".join(stemmed_tokens)

    def _format_price(self, value):
        """Formata valores numéricos no padrão monetário brasileiro simples."""
        try:
            return f"R${float(value):.2f}".replace('.', ',')
        except (TypeError, ValueError):
            return ""

    def _get_random_response(self, tag):
        """Obtém uma resposta aleatória de um intent específico."""
        for intent in self.intents:
            if intent.get('tag') == tag and intent.get('responses'):
                return random.choice(intent['responses'])
        return ""

    def _compose_multi_intent_answer(self, categories_found):
        """Monta uma resposta única com base nas categorias identificadas."""
        parts = []

        burgers = self.menu.get('hamburgueres', [])
        if burgers and (categories_found.get('preço') or categories_found.get('cardápio')):
            burger_details = []
            for item in burgers:
                name = item.get('nome')
                price = self._format_price(item.get('preco'))
                if name and price:
                    burger_details.append(f"{name} ({price})")
            if burger_details:
                parts.append(f"Nossos hambúrgueres: {', '.join(burger_details)}.")

        bebidas = self.menu.get('bebidas', [])
        if bebidas and (categories_found.get('bebida') or categories_found.get('preço')):
            refrigerantes = [b for b in bebidas if b.get('tipo', '').lower() == 'refrigerante']
            if refrigerantes:
                refri = refrigerantes[0]
                refri_price = self._format_price(refri.get('preco'))
                if refri.get('nome') and refri_price:
                    parts.append(f"O refrigerante {refri.get('nome')} custa {refri_price}.")

            other_beverages = [b for b in bebidas if b.get('tipo', '').lower() != 'refrigerante']
            if other_beverages:
                beverage_texts = []
                for beverage in other_beverages:
                    price = self._format_price(beverage.get('preco'))
                    name = beverage.get('nome')
                    beverage_type = beverage.get('tipo', '')
                    if not name or not price:
                        continue
                    display_name = name
                    if beverage_type and beverage_type.lower() not in name.lower():
                        display_name = f"{beverage_type} {name}"
                    beverage_texts.append(f"{display_name} ({price})")
                if beverage_texts:
                    parts.append(f"Também servimos {', '.join(beverage_texts)}.")

        sides = self.menu.get('acompanhamentos', [])
        if sides and categories_found.get('cardápio'):
            sides_details = []
            for item in sides:
                name = item.get('nome')
                price = self._format_price(item.get('preco'))
                if name and price:
                    sides_details.append(f"{name} ({price})")
            if sides_details:
                parts.append(f"Para acompanhar, temos {', '.join(sides_details)}.")

        if categories_found.get('entrega'):
            delivery_text = self._get_random_response('tempo_entrega')
            if not delivery_text:
                delivery_text = "Nosso tempo médio de entrega fica entre 35 e 50 minutos, conforme sua região."
            parts.append(delivery_text)

        return " ".join(parts).strip()

    def train(self):
        """
        Prepara os dados, treina o pipeline (vetorizador + classificador) e o salva.
        """
        print("Iniciando o treinamento do modelo...")

        # Preparar dados de treino
        documents = []
        tags = []
        for intent in self.intents:
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
            combined_answer = self._compose_multi_intent_answer(categories_found)
            if combined_answer:
                return "multiplas_intencoes", 0.97, combined_answer

            # Primeiro tenta identificar intenções específicas para cada categoria
            responses = []

            # Predição principal (geral)
            prediction_proba = self.model.predict_proba([processed_text])
            best_class_index = np.argmax(prediction_proba[0])
            main_intent = self.model.classes_[best_class_index]
            main_confidence = prediction_proba[0][best_class_index]

            # Se a intenção principal for clara (alta confiança), usa ela
            if main_confidence > 0.7:
                answer = self._get_random_response(main_intent)
                if answer:
                    return main_intent, main_confidence, answer

            # Caso contrário, tenta responder cada parte separadamente
            intent_responses = []

            # Verificar cada categoria e buscar respostas específicas
            if categories_found.get('preço'):
                preco_answer = self._get_random_response('ver_preco')
                if preco_answer:
                    intent_responses.append(preco_answer)

            if categories_found.get('cardápio'):
                cardapio_answer = self._get_random_response('ver_cardapio_geral')
                if cardapio_answer:
                    intent_responses.append(cardapio_answer)

            if categories_found.get('entrega'):
                entrega_answer = self._get_random_response('tempo_entrega')
                if entrega_answer:
                    intent_responses.append(entrega_answer)

            # Se encontrou respostas específicas, combina-as
            if intent_responses:
                combined_answer = " ".join(intent_responses)
                return "multiplas_intencoes", 0.95, combined_answer

            # Se não conseguiu responder especificamente, usa a resposta genérica de múltiplas intenções
            generic_answer = self._get_random_response('multiplas_intencoes')
            if generic_answer:
                return "multiplas_intencoes", 0.95, generic_answer
        
        # Caso contrário, segue com a predição normal
        prediction_proba = self.model.predict_proba([processed_text])
        
        # Pega a melhor classe e sua probabilidade
        best_class_index = np.argmax(prediction_proba[0])
        intent_tag = self.model.classes_[best_class_index]
        confidence = prediction_proba[0][best_class_index]
        
        # Selecionar uma resposta
        answer = "Desculpe, não entendi. Pode reformular?"
        for intent in self.intents:
            if intent['tag'] == intent_tag:
                answer = random.choice(intent['responses'])
                break
        
        return intent_tag, confidence, answer

if __name__ == '__main__':
    # Bloco para executar o treinamento diretamente
    nlp_model = NLPModel()
    nlp_model.train()

