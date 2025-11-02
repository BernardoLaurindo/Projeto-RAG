import streamlit as st
import pandas as pd
import os
import re
import json
import traceback
from openai import OpenAI
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "data/produtos.csv"
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
client = None
if OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)

def carregar_base():
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df = pd.DataFrame(columns=[
            "product_id","seller_id","category","brand","price","seller_rating",
            "seller_reviews","product_images","description_length","shipping_time_days"
        ])
        df.to_csv(DATA_PATH, index=False)
        return df
    return pd.read_csv(DATA_PATH, dtype={
        "product_id": str, "seller_id": str, "category": str, "brand": str
    })

def salvar_base(df):
    df.to_csv(DATA_PATH, index=False)

@st.cache_resource
def criar_embeddings_obj():
    return OpenAIEmbeddings()

@st.cache_resource
def criar_vectorstore(df):
    docs = []
    for _, row in df.iterrows():
        texto = (
            f"ID: {row['product_id']} | Categoria: {row['category']} | Marca: {row['brand']} | "
            f"Preço: {row['price']} USD | Seller: {row['seller_id']} | Rating: {row['seller_rating']} | "
            f"Imagens: {row['product_images']} | Desc_size: {row['description_length']} | "
            f"Envio: {row['shipping_time_days']} dias"
        )
        docs.append(Document(page_content=texto))

    if not docs:
        return None

    embeddings = criar_embeddings_obj()
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

def calcular_risco_local(product_id, df):
    prod_row = df[df["product_id"] == str(product_id)]
    if prod_row.empty:
        return {"ok": False, "error": "Produto não encontrado"}

    p = prod_row.iloc[0]
    motivos = []
    score = 0.0

    avg_price = df["price"].astype(float).mean()
    price = float(p["price"])

    if price < 0.6 * avg_price:
        motivos.append("Preço muito abaixo da média")
        score += 1.5
    if price < 0.8 * avg_price:
        motivos.append("Preço abaixo da média")
        score += 0.5
    if float(p["seller_rating"]) < 3.5:
        motivos.append("Rating do vendedor baixo")
        score += 1.0
    if int(p["product_images"]) < 3:
        motivos.append("Poucas imagens")
        score += 0.7
    if int(p["description_length"]) < 100:
        motivos.append("Descrição curta")
        score += 0.7
    if int(p["shipping_time_days"]) > 20:
        motivos.append("Tempo de entrega longo")
        score += 0.6

    if re.search(r"(.)\1\1", str(p["brand"]), re.IGNORECASE):
        motivos.append("Marca com ortografia suspeita")
        score += 0.4

    normalized = min(round((score / 5.0) * 5, 2), 5.0)
    risk = "Low" if normalized < 2 else "Medium" if normalized < 3.5 else "High"

    return {
        "ok": True,
        "score": normalized,
        "risk_level": risk,
        "reasons": motivos
    }

def adicionar_produto_local(payload, df):
    required = [
        "product_id","seller_id","category","brand","price","seller_rating",
        "seller_reviews","product_images","description_length","shipping_time_days"
    ]
    for r in required:
        if payload.get(r) in [None, ""]:
            return {"ok": False, "error": f"Campo obrigatório ausente: {r}"}

    if (df["product_id"] == payload["product_id"]).any():
        return {"ok": False, "error": "product_id já existe"}

    df.loc[len(df)] = payload
    salvar_base(df)
    
    try: criar_vectorstore.clear()
    except: pass

    return {"ok": True, "message": "Produto adicionado com sucesso!"}

def resumo_vendedor_local(seller_id, df):
    dados = df[df["seller_id"] == str(seller_id)]
    if dados.empty:
        return {"ok": False, "error": "Vendedor não encontrado"}

    return {
        "ok": True,
        "produto_count": len(dados),
        "rating_mean": round(dados["seller_rating"].astype(float).mean(), 2),
        "price_mean": round(dados["price"].astype(float).mean(), 2),
        "sample_products": dados.head(5).to_dict(orient="records")
    }

def search_products_local(query, df, vectorstore, k=5):
    results = []
    if vectorstore:
        docs = vectorstore.similarity_search(query, k=k)
        for d in docs:
            pid = re.search(r"ID:\s*(.+?)\s*\|", d.page_content)
            if pid:
                pid = pid.group(1).strip()
                row = df[df["product_id"] == pid]
                if not row.empty:
                    results.append(row.iloc[0].to_dict())
    return {"ok": True, "results": results}

FUNCTIONS = [
    {
        "name": "calcular_risco",
        "parameters": {
            "type": "object",
            "properties": {"product_id": {"type": "string"}},
            "required": ["product_id"]
        }
    },
    {
        "name": "adicionar_produto",
        "parameters": {"type": "object"}
    },
    {
        "name": "resumo_vendedor",
        "parameters": {
            "type": "object",
            "properties": {"seller_id": {"type": "string"}},
            "required": ["seller_id"]
        }
    },
    {
        "name": "buscar_produtos",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer"}}
        }
    }
]

def gerar_resposta_via_agent(user_message, df, vectorstore):
    if client is None:
        return {"ok": False, "error": "OPENAI_API_KEY não configurada"}

    messages = [
        {"role": "system", "content": "Você é um agente anti-fraude especializado."},
        {"role": "user", "content": user_message}
    ]

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            functions=FUNCTIONS,
            function_call="auto"
        )
        msg = resp.choices[0].message

        if msg.get("function_call"):
            name = msg["function_call"]["name"]
            args = json.loads(msg["function_call"]["arguments"])
            
            if name == "calcular_risco":
                result = calcular_risco_local(args["product_id"], df)
            elif name == "adicionar_produto":
                result = adicionar_produto_local(args, df)
            elif name == "resumo_vendedor":
                result = resumo_vendedor_local(args["seller_id"], df)
            elif name == "buscar_produtos":
                result = search_products_local(args["query"], df, vectorstore)
            else:
                result = {"ok": False, "error": "Função desconhecida"}

            messages.append({
                "role": "function",
                "name": name,
                "content": json.dumps(result)
            })

            follow = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return {"ok": True, "assistant": follow.choices[0].message.content}
        else:
            return {"ok": True, "assistant": msg.get("content")}
    
    except Exception as e:
        return {"ok": False, "error": str(e)}

def configurar_pagina():
    st.set_page_config(page_title="MarketGuard - IA Anti-Fraude", layout="wide")
    st.title("🛡️ MarketGuard — IA Anti-Fraude")
    st.caption("Detecte e investigue produtos suspeitos no marketplace")

def main():
    configurar_pagina()
    df = carregar_base()

    try: vectorstore = criar_vectorstore(df)
    except: vectorstore = None

    tab_chat, tab_manual = st.tabs(["Chat com o Agente", "Ferramentas Manuais"])

    with tab_chat:
        st.subheader("Assistente Anti-Fraude Inteligente")

        if "chat" not in st.session_state:
            st.session_state.chat = []

        for m in st.session_state.chat:
            st.markdown(f"**{m['role'].capitalize()}:** {m['content']}")

        msg = st.text_input("Envie sua consulta", key="chat_input")
        if st.button("Enviar") and msg:
            st.session_state.chat.append({"role": "user", "content": msg})
            out = gerar_resposta_via_agent(msg, df, vectorstore)

            if out["ok"]:
                st.session_state.chat.append({"role": "assistant", "content": out["assistant"]})
            else:
                st.error(out["error"])

            st.experimental_rerun()

    # MANUAL
    with tab_manual:
        st.subheader("Ferramentas Manuais")

        st.write(f"Produtos cadastrados: **{len(df)}**")

        st.markdown("### Analisar Risco")
        pid = st.text_input("Product ID")
        if st.button("Calcular Risco"):
            res = calcular_risco_local(pid, df)
            if res["ok"]:
                st.metric("Risco", res["risk_level"])
                for r in res["reasons"]:
                    st.write("- ", r)
            else:
                st.error(res["error"])

        st.markdown("---")
        st.markdown("### Cadastrar Produto")
        novo = {}
        for campo in df.columns:
            novo[campo] = st.text_input(campo)

        if st.button("Cadastrar"):
            out = adicionar_produto_local(novo, df)
            if out["ok"]: st.success(out["message"])
            else: st.error(out["error"])

        st.markdown("---")
        st.markdown("### Resumo do Vendedor")
        sid = st.text_input("Seller ID")
        if st.button("Resumo"):
            res = resumo_vendedor_local(sid, df)
            if res["ok"]:
                st.write(res)
            else:
                st.error(res["error"])

        st.markdown("---")
        if st.button("Reindexar Base Vetorial"):
            try:
                criar_vectorstore.clear()
                st.success("Vectorstore será recriado no próximo uso")
            except:
                st.error("Erro ao limpar cache")

if __name__ == "__main__":
    main()
