# 🛡️ MarketGuard — IA Anti-Fraude para Marketplaces

O MarketGuard é um sistema inteligente que identifica potenciais produtos falsificados em plataformas de e-commerce.  
Ele utiliza **RAG (Retrieval-Augmented Generation)** com **Function Calling**, permitindo consultas inteligentes em linguagem natural, análise automática de risco e inclusão de novos produtos na base de conhecimento.

---

## Problema

Marketplaces sofrem com produtos falsificados que:
- usam marcas com pequenos erros ortográficos
- possuem **preços muito abaixo do normal**
- são vendidos por **vendedores com reputação ruim**
- têm **poucas imagens** e **descrições curtas**

Atualmente, auditorias são manuais, lentas e não escalam com o volume de listagens.

---

## Solução

O MarketGuard possibilita que analistas conversem com a IA e:

✅ Consultem produtos suspeitos via **RAG**  
✅ Calculem automaticamente o **score de risco**  
✅ Adicionem novos produtos reportados por usuários  
✅ Investigam histórico e métricas de vendedores específicos  

Tudo diretamente através de **conversação em linguagem natural**.

---

## Tecnologias Utilizadas

| Área | Ferramentas |
|------|-------------|
| Interface | Streamlit |
| IA Conversacional | OpenAI GPT-4o-mini + Function Calling |
| Busca Semântica | LangChain + ChromaDB |
| Base de Dados | CSV + Pandas |
| Processamento | Python 3.10+ |

---

## Estrutura do Projeto

📦 Projeto-RAG

┣ 📁 data/

┃ ┗ produtos.csv

┣ 📄 app.py

┣ 📄 README.md

┗ 📄 requirements.txt


---

## Instalação e Execução

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/BernardoLaurindo/Projeto-RAG.git
cd MarketGuard
```


### 2️⃣ Criar ambiente virtual (opcional, mas recomendado)
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 3️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

### 4️⃣ Executar o sistema
```bash
streamlit run app.py
```
---

## Como Usar

### Chat com o Agente

Exemplos de comandos:

- "Quais produtos suspeitos na categoria Electronics?"

- "Calcule o risco do produto PROD0156"

- "Adicionar novo produto: iPhone — preço 250, seller seller_9921 etc."

- "Resumo do vendedor seller_4521"


### Ferramentas Manuais

Caso a API não esteja configurada:

- Análise de risco por ID

- Cadastro manual de produto

- Resumo de vendedor

- Reindexação da base vetorial