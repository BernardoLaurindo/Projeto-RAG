import streamlit as st
import os
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Configurações e API key
os.environ["OPENAI_API_KEY"] = "sua_api_key_aqui"

def configurar_pagina():
    st.set_page_config(
        page_title="MarketGuard - Detecção de Fraudes",
        page_icon="🔍",
        layout="wide"
    )
    st.title("MarketGuard - Detecção Inteligente de Produtos Falsificados")
    st.markdown("""
    Este sistema usa RAG e funções específicas para ajudar na análise de risco e cadastro de produtos suspeitos.
    """)

# Carregamento base dados
@st.cache_data
def carregar_dados():
    produtos = pd.read_csv('data/produtos.csv')
    try:
        with open('data/vector_db.pkl', 'rb') as f:
            vector_db = pickle.load(f)
    except:
        vector_db = {}
    return produtos, vector_db

def salvar_base_vetorial(vector_db):
    with open('data/vector_db.pkl', 'wb') as f:
        pickle.dump(vector_db, f)

# Embeddings com sentence-transformers
def gerar_embedding(texto):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode([texto])[0]

# RAG - busca por similaridade (simplificada)
def buscar_produtos_similares(query, produtos, vector_db):
    query_emb = gerar_embedding(query)
    # Aqui idealmente usar FAISS ou similar - exemplo mock
    resultados = produtos.sample(5)  # Mock - implemente busca vetorial real
    return resultados

# Análise de risco
def calcular_risco(produto_id, produtos):
    prod = produtos[produtos['product_id'] == produto_id]
    if prod.empty:
        return f"Produto {produto_id} não encontrado."
    
    prod = prod.iloc[0]
    score = 0
    detalhes = []
    
    # Exemplos simples de regras
    avg_price = produtos['price'].mean()
    if prod.price < 0.6 * avg_price:
        score += 1
        detalhes.append("Preço muito abaixo da média")
    if prod.seller_rating < 3.5:
        score += 1
        detalhes.append("Rating do vendedor baixo")
    if prod.product_images < 3:
        score += 1
        detalhes.append("Poucas imagens")
    if prod.description_length < 100:
        score += 1
        detalhes.append("Descrição muito curta")
    if prod.shipping_time_days > 20:
        score += 1
        detalhes.append("Tempo de entrega longo")
    # Simples checagem de erros em marca pode ser implementada adicionalmente

    return f"Score de risco: {score}/5\nMotivos: {', '.join(detalhes)}"

# Cadastro de produto novo
def cadastrar_produto(dados_texto, produtos, vector_db):
    # Exemplo simplificado: extração manual/parsing pode ser expandido
    # Espera-se string formatada: "produto_id, seller_id, category, brand, price, seller_rating, seller_reviews, product_images, description_length, shipping_time_days"
    try:
        campos = dados_texto.split(',')
        novo = {
            'product_id': campos[0].strip(),
            'seller_id': campos[1].strip(),
            'category': campos[2].strip(),
            'brand': campos[3].strip(),
            'price': float(campos[4]),
            'seller_rating': float(campos[5]),
            'seller_reviews': int(campos[6]),
            'product_images': int(campos[7]),
            'description_length': int(campos[8]),
            'shipping_time_days': int(campos[9])
        }
    except Exception as e:
        return f"Erro no formato de cadastro: {e}"
    
    produtos = produtos.append(novo, ignore_index=True)
    produtos.to_csv('data/produtos.csv', index=False)
    
    # Atualizar vetor DB
    texto_emb = f"{novo['brand']} {novo['category']} {novo['seller_id']} {novo['price']}"
    emb = gerar_embedding(texto_emb)
    vector_db[novo['product_id']] = emb
    salvar_base_vetorial(vector_db)

    return "Produto cadastrado e base vetorial atualizada."

# Estatísticas do vendedor
def resumo_vendedor(vendedor_id, produtos):
    dados = produtos[produtos['seller_id'] == vendedor_id]
    if dados.empty:
        return f"Nenhum produto encontrado para vendedor {vendedor_id}."
    resumo = {
        "Produtos cadastrados": len(dados),
        "Rating médio": dados['seller_rating'].mean(),
        "Preço médio dos produtos": dados['price'].mean(),
        "Produtos suspeitos com preço abaixo de 60% da média": dados.query('price < (0.6 * price.mean())').shape[0]
    }
    return resumo

# Interface principal
def main():
    configurar_pagina()
    produtos, vector_db = carregar_dados()

    pergunta = st.text_area("Digite sua consulta ou solicitação:")
    if st.button("Enviar"):
        if "risco" in pergunta.lower():
            prod_id = pergunta.split()[-1].strip()
            st.write(calcular_risco(prod_id, produtos))
        elif "cadastrar" in pergunta.lower():
            # O usuário deve enviar dados separados por vírgula
            st.write(cadastrar_produto(pergunta.replace("cadastrar","").strip(), produtos, vector_db))
        elif "vendedor" in pergunta.lower():
            vendedor_id = pergunta.split()[-1].strip()
            st.json(resumo_vendedor(vendedor_id, produtos))
        else:
            resultados = buscar_produtos_similares(pergunta, produtos, vector_db)
            st.dataframe(resultados)

if __name__ == "__main__":
    main()
