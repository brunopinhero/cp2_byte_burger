# -*- coding: utf-8 -*-

import nltk
import re
from flask import Flask, render_template, request, jsonify
from nlp_core import NLPModel

# --- Configuração da Aplicação ---
app = Flask(__name__)
# Constante para limitar o número de sentenças por requisição
MAX_SENTENCES = 10

# --- Carregamento do Modelo de NLP ---
# Tenta carregar o modelo treinado ao iniciar a aplicação.
# Se falhar, a variável 'nlp' ficará como None e um erro será retornado pela API.
try:
    nlp = NLPModel()
    nlp.load_model()
except Exception as e:
    print(f"ERRO CRÍTICO: Não foi possível carregar o modelo de NLP. Detalhes: {e}")
    print("Por favor, execute 'python nlp_core.py' para treinar e gerar os arquivos do modelo.")
    nlp = None

# --- Definição das Rotas ---

@app.route('/')
def index():
    """
    Rota principal que renderiza a interface do chat (o arquivo index.html).
    """
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint principal da API. Recebe uma mensagem do usuário, processa cada sentença
    e retorna uma lista de respostas correspondentes.
    """
    # Verifica se o modelo foi carregado corretamente na inicialização.
    if not nlp:
        return jsonify({"error": "O modelo de NLP não está carregado. Verifique o console do servidor para mais detalhes."}), 500

    # Extrai a mensagem do corpo JSON da requisição.
    data = request.get_json()
    user_message = data.get('message', '')

    # Se a mensagem estiver vazia, retorna uma lista de respostas vazia.
    if not user_message.strip():
        return jsonify({"responses": []})

    # Verifica se a mensagem pode conter múltiplas perguntas através de padrões comuns
    # como perguntas sobre cardápio, preço, tempo de entrega, etc.
    keywords = {
        'preço': ['preço', 'preco', 'valor', 'custa', 'custo', 'quanto'],
        'cardápio': ['cardápio', 'cardapio', 'menu', 'opções', 'opcoes', 'lanches', 'hamburgueres', 'hamburgers'],
        'tempo': ['entrega', 'demora', 'tempo', 'frete', 'delivery', 'quanto tempo']
    }
    
    # Conta quantas categorias de palavras-chave estão presentes na mensagem
    categories_found = sum(1 for category, words in keywords.items() 
                          if any(word in user_message.lower() for word in words))
    
    # Se encontrar mais de uma categoria, considera como múltiplas intenções
    is_multiple_intents = categories_found > 1
    
    if is_multiple_intents:
        # Para múltiplas intenções, usamos o predict_intent uma única vez
        try:
            intent, probability, answer = nlp.predict_intent(user_message)
            response_obj = {
                'intent': intent,
                'probability': float(probability),
                'answer': answer
            }
            return jsonify({"responses": [response_obj]})
        except Exception as e:
            print(f"Erro ao processar múltiplas intenções: '{user_message}'. Detalhes: {e}")
            return jsonify({"responses": [{
                'intent': 'erro',
                'probability': 0.0,
                'answer': 'Desculpe, tive um problema ao processar sua pergunta. Pode reformular?'
            }]})
    
    # Se não for múltipla intenção, processa cada sentença separadamente
    try:
        sentences = nltk.sent_tokenize(user_message, language='portuguese')
    except Exception as e:
        # Em caso de erro na tokenização, trata a mensagem como uma única sentença.
        print(f"Alerta: Falha na tokenização NLTK. Tratando como uma única sentença. Erro: {e}")
        sentences = [user_message]
        
    responses = []
    # Itera sobre cada sentença (respeitando o limite máximo).
    for sentence in sentences[:MAX_SENTENCES]:
        # Ignora sentenças que são apenas espaços em branco.
        if sentence.strip():
            try:
                # Prevê a intenção para a sentença atual.
                intent, probability, answer = nlp.predict_intent(sentence)
                
                # Monta o objeto de resposta.
                response_obj = {
                    'intent': intent,
                    'probability': float(probability), # Garante que o tipo é float.
                    'answer': answer
                }
                responses.append(response_obj)
            except Exception as e:
                # Se a predição para uma sentença específica falhar, registra o erro e continua.
                print(f"Erro ao processar a sentença: '{sentence}'. Detalhes: {e}")

    # Retorna a lista de respostas em formato JSON.
    return jsonify({"responses": responses})

# --- Execução da Aplicação ---
if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "5000"))  # Render injeta $PORT
    app.run(host="0.0.0.0", port=port, debug=False)


