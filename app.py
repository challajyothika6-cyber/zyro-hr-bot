import os
import glob
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

MODEL = "openai/gpt-oss-120b"

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="💼")
st.title("💼 Zyro Dynamics HR Help Desk")
st.caption("Ask any HR policy question. Answers come only from official policy documents.")

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

REFUSAL = (
    "I'm sorry, but I can only answer questions about Zyro Dynamics HR policies, "
    "and I couldn't find this in the HR documents. Please contact the HR team directly."
)

PROMPT = ChatPromptTemplate.from_template(
    """You are the HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer the employee's question using ONLY the context below.

Rules:
- Questions about leave, salary, benefits, insurance, notice period, travel allowances, performance,
  work from home, IT policy, conduct, POSH, onboarding and separation are HR policy questions.
  If the context has relevant information, ANSWER with the exact numbers, days, limits and conditions.
- Refuse ONLY if the question is about something not covered by HR policies, such as company revenue
  or finances, products, other companies or competitors, applying for jobs, or general knowledge.
  In that case reply with exactly: "{refusal}"
- Never use outside knowledge and never guess.

Context:
{context}

Question: {question}

Answer:"""
).partial(refusal=REFUSAL)


@st.cache_resource(show_spinner="Loading HR policies...")
def build_pipeline():
    docs = []
    for p in sorted(set(glob.glob("data/*.pdf") + glob.glob("*.pdf"))):
        pages = PyPDFLoader(p).load()
        for pg in pages:
            pg.metadata["source"] = os.path.basename(p)
        docs.extend(pages)
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_documents(docs)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.from_documents(chunks, emb)
    retriever = vs.as_retriever(
        search_type="mmr", search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.7}
    )
    llm = ChatGroq(model=MODEL, temperature=0)
    return retriever, PROMPT | llm | StrOutputParser()


retriever, chain = build_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.write("•", s)

if question := st.chat_input("e.g. How many casual leaves do I get?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            answer = chain.invoke({"context": context, "question": question})
            sources = [] if answer.strip() == REFUSAL else sorted(
                {f"{d.metadata['source']} (page {d.metadata.get('page', 0) + 1})" for d in docs}
            )
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.write("•", s)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
