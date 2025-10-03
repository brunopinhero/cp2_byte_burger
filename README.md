# Byte Burger Chatbot

## 📋 Descrição

O Byte Burger Chatbot é uma aplicação de inteligência artificial que utiliza processamento de linguagem natural (NLP) para simular uma conversa com clientes de uma hamburgueria. O sistema interpreta as perguntas dos usuários, identifica suas intenções e fornece respostas adequadas sobre o cardápio, horários de funcionamento, promoções e outras informações relevantes.

## 🚀 Tecnologias Utilizadas

- **Python**: Linguagem de programação principal
- **Flask**: Framework web para o backend
- **NLTK**: Biblioteca para processamento de linguagem natural
- **Scikit-learn**: Biblioteca para machine learning (LogisticRegression, TF-IDF)
- **HTML/CSS/JavaScript**: Frontend para interface do usuário

## 🧠 Recursos Avançados de NLP

- **Detecção de múltiplas intenções**: Capaz de identificar e responder a várias perguntas em uma única mensagem
- **Extração de entidades**: Reconhece itens específicos do cardápio mencionados nas perguntas
- **Normalização de texto**: Remove acentos e caracteres especiais para melhorar a precisão da busca
- **Stemming**: Reduz palavras às suas raízes para melhor comparação
- **Remoção de stopwords**: Elimina palavras comuns que não agregam significado
- **Indexação de aliases**: Sistema de apelidos para os itens do cardápio para melhor reconhecimento

## 📦 Estrutura do Projeto

```
byte_burger/
├── app.py                # Aplicação principal Flask com rotas e processamento de requisições
├── database.json         # Base de dados com intenções, respostas e cardápio detalhado
├── nlp_core.py           # Núcleo de processamento de linguagem natural e machine learning
├── requirements.txt      # Dependências do projeto
├── modelo_pipeline.pkl   # Modelo NLP treinado (gerado automaticamente)
├── static/               # Arquivos estáticos
│   ├── css/
│   │   └── styles.css    # Estilos da interface
│   └── js/
│       └── script.js     # Scripts do frontend para gerenciamento do chat
└── templates/
    └── index.html        # Template da página principal do chat
```

## ⚙️ Pré-requisitos

- Python 3.8 ou superior
- Pip (gerenciador de pacotes do Python)
- Acesso à internet (para download inicial dos recursos do NLTK)

## 🔧 Instalação e Configuração

### Windows

1. **Instalar Python**:

   - Baixe e instale o Python do [site oficial](https://www.python.org/downloads/windows/)
   - Durante a instalação, marque a opção "Add Python to PATH"

2. **Clonar o repositório** (opcional - se estiver em um repositório Git):

   ```cmd
   git clone <url-do-repositório>
   cd byte-burger
   ```

3. **Configurar ambiente virtual** (recomendado):

   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Instalar dependências**:

   ```cmd
   pip install -r requirements.txt
   ```

5. **Treinar o modelo NLP**:
   ```cmd
   python nlp_core.py
   ```

### macOS

1. **Instalar Python** (se não estiver instalado):

   - Instale o Homebrew (se não tiver):
     ```bash
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```
   - Instale o Python:
     ```bash
     brew install python
     ```

2. **Clonar o repositório** (opcional):

   ```bash
   git clone <url-do-repositório>
   cd byte-burger
   ```

3. **Configurar ambiente virtual** (recomendado):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Instalar dependências**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Treinar o modelo NLP**:
   ```bash
   python nlp_core.py
   ```

### Linux (Ubuntu/Debian)

1. **Atualizar e instalar Python**:

   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv -y
   ```

2. **Clonar o repositório** (opcional):

   ```bash
   git clone <url-do-repositório>
   cd byte-burger
   ```

3. **Configurar ambiente virtual** (recomendado):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Instalar dependências**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Treinar o modelo NLP**:
   ```bash
   python nlp_core.py
   ```

## 🚀 Executando a Aplicação

Depois de completar a instalação e o treinamento do modelo, você pode executar a aplicação:

### Windows

```cmd
# Certifique-se que o ambiente virtual está ativado
venv\Scripts\activate
# Execute a aplicação
python app.py
```

### macOS e Linux

```bash
# Certifique-se que o ambiente virtual está ativado
source venv/bin/activate
# Execute a aplicação
python app.py
```

O servidor será iniciado e estará disponível em: http://127.0.0.1:5000

## 🔍 Uso da Aplicação

1. Abra seu navegador e acesse: http://127.0.0.1:5000
2. Use a interface de chat para interagir com o Byte Burger Chatbot
3. Digite suas perguntas naturalmente, como:
   - "Qual é o cardápio?"
   - "Quanto custa o hambúrguer Classic Geek?"
   - "Vocês têm opções vegetarianas?"
   - "Quais bebidas vocês têm e quanto custam?"
   - "Quanto tempo demora para entregar e qual é o preço do Veggie Kernel?"

## 🧪 Recursos Avançados

### Detecção de Múltiplas Intenções

O chatbot é capaz de identificar e responder a várias perguntas em uma única mensagem. Por exemplo, se o usuário perguntar "Quais são os preços dos lanches e quanto tempo demora a entrega?", o sistema reconhece ambas as intenções e fornece uma resposta completa.

### Reconhecimento de Itens Específicos

O sistema identifica menções a itens específicos do cardápio, mesmo usando apelidos ou variações de nomes. Por exemplo, "Quanto custa o Classic?" será entendido como uma pergunta sobre o "Classic Geek Burger".

### Processamento de Linguagem Natural

- **Normalização de texto**: Remove acentos e caracteres especiais
- **Stemming**: Reduz palavras às suas raízes (ex: "hambúrgueres" → "hamburg")
- **Remoção de stopwords**: Elimina palavras comuns que não agregam significado
- **Vetorização TF-IDF**: Converte texto em representações numéricas com base na importância das palavras
- **Modelo de Classificação**: Usa regressão logística para classificar a intenção do usuário

## 🛠️ Possíveis Problemas e Soluções

### Erro ao baixar recursos NLTK

Se ocorrer um erro ao baixar os recursos do NLTK, você pode baixá-los manualmente:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')
```

### Erro ao carregar o modelo

Se aparecer a mensagem "O modelo de NLP não está carregado", certifique-se de executar `python nlp_core.py` para treinar o modelo antes de iniciar a aplicação.

### Problemas com dependências

Em caso de erros relacionados a dependências, tente atualizar as bibliotecas:

```bash
pip install --upgrade -r requirements.txt
```

## 🔄 Personalizando o Chatbot

Para personalizar as intenções, respostas e o cardápio do chatbot, edite o arquivo `database.json`. Este arquivo contém:

1. **Intenções e Respostas**: Define os padrões de perguntas e as possíveis respostas para cada intenção:

```json
{
  "intents": [
    {
      "tag": "saudacao",
      "patterns": ["Olá", "Oi", "E aí", "Tudo bem?"],
      "responses": ["Olá! Bem-vindo ao Byte Burger!", "Oi! Como posso ajudar?"]
    },
    ...
  ]
}
```

2. **Cardápio Detalhado**: Informações sobre hambúrgueres, bebidas, acompanhamentos e sobremesas:

```json
{
  "cardapio_detalhado": {
    "hamburgueres": [
      {
        "nome": "Classic Geek",
        "preco": 32.90,
        "descricao": "Hambúrguer clássico com carne, queijo cheddar e alface"
      },
      ...
    ]
  }
}
```

Após modificar o arquivo de base de dados, você precisa treinar o modelo novamente com `python nlp_core.py`.

## 📊 Métricas de Desempenho

Durante o treinamento, o sistema exibe métricas de desempenho do modelo, incluindo:

- **Acurácia**: Porcentagem de previsões corretas nos dados de teste
- **Precisão**: Proporção de identificações positivas corretas
- **Recall**: Proporção de positivos reais identificados corretamente
- **F1-Score**: Média harmônica entre precisão e recall

## 👥 Autor

- Bruno Pinheiro - [brunopinhero@gmail.com]

---

Projeto de IA e Machine Learning (FIAP) - 2025
