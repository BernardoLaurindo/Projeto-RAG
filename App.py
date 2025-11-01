import streamlit as st
import pandas as pd
import os
import time
import re
from openai import OpenAI
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


os.environ["OPENAI_API_KEY"] = "SUA_API_KEY_AQUI"

DATA_PATH = "data/produtos.csv"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================
# CONFIG STREAMLIT
# ============================
def configurar_pagina():
    st.set_page_config(
        page_title="MarketGuard – Anti-Fraude",
        page_icon="🛡️",
        layout="wide"
    )
    st.title("🛡️ MarketGuard – Detecção Inteligente de Produtos Falsificados")
    st.markdown("""
    Plataforma para análise conversacional com RAG para ajudar analistas
    na detecção de produtos suspeitos de falsificação.
    """)

# ============================
# CARREGAMENTO E PREPARO DA BASE
# ============================
def carregar_base():
    if not os.path.exists(DATA_PATH):
        df = pd.DataFrame(columns=[
            "product_id","seller_id","category","brand","price","seller_rating",
            "seller_reviews","product_images","description_length","shipping_time_days"
        ])
        df.to_csv(DATA_PATH, index=False)
        return df
    return pd.read_csv(DATA_PATH)

def criar_documentos_produtos(df):
    docs = []
    for _, row in df.iterrows():
        texto = (
            f"ID: {row['product_id']} | Categoria: {row['category']} | Marca: {row['brand']} | "
            f"Preço: {row['price']} USD | Seller: {row['seller_id']} | Rating: {row['seller_rating']} | "
            f"Imagens: {row['product_images']} | Desc_size: {row['description_length']} | "
            f"Envio: {row['shipping_time_days']} dias"
        )
        docs.append(Document(page_content=texto))
    return docs

# ============================
# BASE VETORIAL CHROMA (RAG)
# ============================
def criar_base_vetorial(df):
    try:
        docs = criar_documentos_produtos(df)
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(docs, embeddings)
        return vectorstore, len(docs)
    except Exception as e:
        st.error(f"Erro ao criar base vetorial: {e}")
        return None, 0

def buscar_rag(pergunta, vectorstore):
    try:
        docs = vectorstore.similarity_search(pergunta, k=3)
        contexto = "\n".join(doc.page_content for doc in docs)

        prompt = f"""
        Você é um analista de conformidade. Use APENAS o contexto para responder.

        CONTEXTO:
        {contexto}

        PERGUNTA: {pergunta}

        Se a informação não estiver no contexto, diga que não foi encontrada.
        """

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content, docs
    except Exception as e:
        st.error(f"Erro no RAG: {e}")
        return None, []

# ============================
# FUNÇÕES DE ANÁLISE EXIGIDAS
# ============================
def calcular_risco(prod):
    motivos = []
    score = 0

    if prod["price"] < 0.6 * df["price"].mean():
        motivos.append("💸 Preço muito baixo")
        score += 1
    if prod["seller_rating"] < 3.5:
        motivos.append("⭐ Rating do vendedor baixo")
        score += 1
    if prod["product_images"] < 3:
        motivos.append("🖼️ Poucas imagens")
        score += 1
    if prod["description_length"] < 100:
        motivos.append("📄 Descrição curta")
        score += 1
    if prod["shipping_time_days"] > 20:
        motivos.append("🚚 Entrega longa")
        score += 1
    if re.search(r'(.)\1\1', prod["brand"], re.IGNORECASE):
        motivos.append("⚠️ Marca com ortografia suspeita")
        score += 1

    return score, motivos

# ============================
# INTERFACE
# ============================
def main():
    configurar_pagina()
    global df
    df = carregar_base()

    if df.empty:
        st.warning("Nenhum produto cadastrado ainda.")
    else:
        st.success(f"Produtos carregados: {len(df)}")

    vectorstore, qtd = criar_base_vetorial(df)
    st.info(f"Base vetorial criada com {qtd} embeddings")

    tab1, tab2, tab3 = st.tabs(["🔍 Consulta RAG", "🚨 Análise de Risco", "➕ Novo Produto"])

    # TAB 1 - RAG
    with tab1:
        st.subheader("Consulta Inteligente")
        q = st.text_input("Ex: 'produtos de luxo suspeitos'")
        if st.button("Buscar"):
            resp, docs = buscar_rag(q, vectorstore)
            if resp:
                st.markdown("### ✅ Resposta:")
                st.write(resp)
                with st.expander("📌 Trechos utilizados:"):
                    for i, d in enumerate(docs):
                        st.write(f"{i+1}️⃣ {d.page_content}")

    # TAB 2 - Risk
    with tab2:
        st.subheader("Analisar risco por product_id")
        pid = st.text_input("Digite o Product ID")
        if st.button("Analisar"):
            prod = df[df["product_id"] == pid]
            if not prod.empty:
                prod = prod.iloc[0]
                score, motivos = calcular_risco(prod)
                st.metric("Score de risco (0-6)", score)
                for m in motivos:
                    st.write("✅", m)
            else:
                st.error("Produto não encontrado")

    # TAB 3 - Cadastro
    with tab3:
        st.subheader("Adicionar produto suspeito")
        col1, col2 = st.columns(2)

        with col1:
            product_id = st.text_input("ID do produto *")
            seller_id = st.text_input("ID do vendedor *")
            category = st.selectbox("Categoria", df["category"].unique().tolist())
            brand = st.text_input("Marca *")

        with col2:
            price = st.number_input("Preço USD", min_value=0.0, value=10.0)
            seller_rating = st.slider("Rating vendedor", 1.0, 5.0, 4.0)
            product_images = st.number_input("Qtd imagens", min_value=0, value=1)
            description_length = st.number_input("Tam. descrição", min_value=0, value=50)
            shipping_time_days = st.number_input("Dias envio", min_value=1, value=25)

        if st.button("Cadastrar"):
            novo = {
                "product_id": product_id,
                "seller_id": seller_id,
                "category": category,
                "brand": brand,
                "price": price,
                "seller_rating": seller_rating,
                "seller_reviews": 1,
                "product_images": product_images,
                "description_length": description_length,
                "shipping_time_days": shipping_time_days,
            }
            df.loc[len(df)] = novo
            df.to_csv(DATA_PATH, index=False)
            st.success("✅ Produto cadastrado e indexado!")

if __name__ == "__main__":
    main()
