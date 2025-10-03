# Byte Burger Chatbot

## 📋 Descrição

O Byte Burger Chatbot é uma aplicação de inteligência artificial que utiliza processamento de linguagem natural (NLP) para simular uma conversa com clientes de uma hamburgueria. O sistema interpreta as perguntas dos usuários, identifica suas intenções e fornece respostas adequadas sobre o cardápio, horários de funcionamento, promoções e outras informações relevantes.

## 🚀 Tecnologias Utilizadas

- **Python**: Linguagem de programação principal
- **Flask**: Framework web para o backend
- **NLTK**: Biblioteca para processamento de linguagem natural
- **Scikit-learn**: Biblioteca para machine learning
- **HTML/CSS/JavaScript**: Frontend para interface do usuário

## 📦 Estrutura do Projeto

```
byte_burger/
├── app.py                # Aplicação principal Flask
├── database.json         # Base de dados de intenções e respostas
├── nlp_core.py           # Núcleo de processamento de linguagem natural
├── requirements.txt      # Dependências do projeto
├── modelo_pipeline.pkl   # Modelo NLP treinado (gerado automaticamente)
├── static/               # Arquivos estáticos
│   ├── css/
│   │   └── styles.css    # Estilos da interface
│   └── js/
│       └── script.js     # Scripts do frontend
└── templates/
    └── index.html        # Template da página principal
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
   - "Vocês têm hambúrgueres vegetarianos?"
   - "Qual o horário de funcionamento?"
   - "Como faço para pedir delivery?"

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

Para personalizar as intenções e respostas do chatbot, edite o arquivo `database.json`. Este arquivo contém um JSON com a seguinte estrutura:

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

Após modificar o arquivo de base de dados, você precisa treinar o modelo novamente com `python nlp_core.py`.

## 👥 Autor

- Bruno Pinheiro - [brunopinhero@gmail.com]

---
Projeto de IA e Machine Learning (FIAP) - 2025
