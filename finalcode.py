import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import faiss
import gradio as gr
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS as LC_FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from typing import List
from langchain.chains.summarize import load_summarize_chain
import re

# ========== Step 1: Load and Preprocess CSV ==========
df = pd.read_csv("D:\\Vs_code\\GenAi\\Ecommerce_chatbot\\first_500_records.csv")
df.dropna(subset=['Product Name', 'Ratings', 'specifications', 'Price'], inplace=True)
df['Price'] = df['Price'].astype(str).replace('[₹,]', '', regex=True)
df['Price'] = df['Price'].str.extract('(\d+)')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['embedding_index'] = range(len(df))

# ========== Step 2: Create Embeddings ==========
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = (df['Product Name'] + " " + df['specifications']).tolist()
embeddings = model.encode(texts, show_progress_bar=True)
metadata = df[['embedding_index', 'Product Name', 'Price', 'specifications', 'site_url', 'Ratings', 'image_url']].to_dict('records')
documents = [Document(page_content=texts[i], metadata=metadata[i]) for i in range(len(texts))]

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = LC_FAISS.from_documents(documents, embedding_function)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# ========== Step 3: LangChain LLM ==========
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")

# ========== Step 4: Helper Functions ==========

def keyword_filter(query: str, documents: List[Document]) -> List[Document]:
    keywords = ['ram', 'storage', 'battery', 'screen', 'ssd', 'i3', 'i5', 'i7', 'ryzen', 'ms office']
    query_lower = query.lower()
    if not any(kw in query_lower for kw in keywords):
        return documents

    filtered_docs = []
    for doc in documents:
        spec = doc.metadata.get('specifications', '').lower()
        if any(kw in spec for kw in keywords if kw in query_lower):
            filtered_docs.append(doc)

    return filtered_docs if filtered_docs else documents


def recommend_laptops(query: str) -> str:
    results = retriever.get_relevant_documents(query)
    results = keyword_filter(query, results)

    if not results:
        return "❌ No matching laptops found. Please refine your query."

    output = "### 💻 Recommended Laptops for Your Query:\n\n"
    for i, doc in enumerate(results[:5], 1):
        meta = doc.metadata
        output += (
            f"**{i}. {meta.get('Product Name')}**\n"
            f"- 💸 Price: ₹{meta.get('Price')}\n"
            f"- ⭐ Rating: {meta.get('Ratings')}\n"
            f"- 🛠 Features: {meta.get('specifications')[:150]}...\n"  # limit spec length
            f"- 🔗 [Buy Now]({meta.get('site_url')})\n"
            f"- 🖼 ![Image]({meta.get('image_url')})\n\n"
        )
    return output


def extract_spec_from_text(spec_text, keywords):
    # Extract spec info by keywords using regex & split by commas or semicolons for better clarity
    spec_text = spec_text.lower()
    specs = re.split(r',|;', spec_text)
    for kw in keywords:
        for part in specs:
            if kw in part:
                return part.strip().capitalize()
    return "-"


def extract_features(query: str) -> str:
    results = retriever.get_relevant_documents(query)
    results = keyword_filter(query, results)
    if not results:
        return "❌ No relevant laptops found."

    output = "### 🛠 Feature Extracted Laptops:\n\n"
    for i, doc in enumerate(results[:5], 1):
        meta = doc.metadata
        output += (
            f"**{i}. {meta.get('Product Name')}**\n"
            f"- 💸 Price: ₹{meta.get('Price')}\n"
            f"- ⭐ Rating: {meta.get('Ratings')}\n"
            f"- 🛠 Features: {meta.get('specifications')[:200]}...\n"
            f"- 🔗 [Buy here]({meta.get('site_url')})\n"
            f"- 🖼 Image: {meta.get('image_url')}\n\n"
        )
    return output


def compare_laptops(query: str) -> str:
    results = retriever.get_relevant_documents(query)
    results = keyword_filter(query, results)

    if len(results) < 2:
        return "❌ Need at least two relevant laptops to compare."

    headers = ["Product", "Price", "RAM", "Storage", "Screen", "Battery", "MS Office"]
    table = "### 🔍 Laptop Comparison\n\n"
    table += "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"]*len(headers)) + " |\n"

    for doc in results[:5]:
        meta = doc.metadata
        specs = meta.get('specifications', '')

        ram = extract_spec_from_text(specs, ['ram'])
        storage = extract_spec_from_text(specs, ['storage', 'ssd', 'hdd'])
        screen = extract_spec_from_text(specs, ['screen size', 'display'])
        battery = extract_spec_from_text(specs, ['battery'])
        office = extract_spec_from_text(specs, ['ms office', 'office'])

        table += (
            f"| {meta['Product Name'][:25]} "
            f"| ₹{meta['Price']} "
            f"| {ram} "
            f"| {storage} "
            f"| {screen} "
            f"| {battery} "
            f"| {office} |\n"
        )
    return f"```markdown\n{table}\n```"  # Use code block for better table rendering in chat UI


# ========== Step 5: LangChain Tools ==========

rag_tool = Tool(
    name="Laptop Recommender",
    func=recommend_laptops,
    description="Recommends multiple laptops based on user query and filters like price, RAM, features."
)

feature_tool = Tool(
    name="Feature Extractor",
    func=extract_features,
    description="Extracts laptop features such as RAM, Storage, Battery, etc."
)

compare_tool = Tool(
    name="Product Comparator",
    func=compare_laptops,
    description="Compares multiple laptops based on important specifications."
)

memory = ConversationBufferMemory(memory_key="chat_history", max_token_limit=5000, return_messages=True)


def summarize_conversation(_: str) -> str:
    chat_history = memory.buffer
    if not chat_history:
        return "❌ No conversation history to summarize."
    summarize_chain = load_summarize_chain(llm=llm, chain_type="stuff")
    # Join messages into a single string for summarization
    full_text = "\n".join([msg.content for msg in chat_history])
    summary = summarize_chain.run([Document(page_content=full_text)])
    return f"📝 **Summary of our conversation so far:**\n\n{summary}"


summary_tool = Tool(
    name="Chat Summary",
    func=summarize_conversation,
    description="Summarizes the conversation so far."
)

# ========== Step 6: LangChain Agent Setup ==========
agent_executor = initialize_agent(
    tools=[rag_tool, feature_tool, compare_tool, summary_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False,  # Set to False for cleaner logs
    memory=memory
)

# ========== Step 7: Gradio Chat UI ==========

qa_history = []

def run_safe_query(query: str):
    forbidden_keywords = ["mobile", "smartphone", "phone", "cellphone"]
    if any(word in query.lower() for word in forbidden_keywords):
        return "❌ I can only help with laptops. Please ask about laptops, not mobile phones."

    try:
        response = agent_executor.invoke({"input": query})
        answer = response["output"]

        # Save history (limit last 5 Q&A)
        qa_history.append((query, answer))
        if len(qa_history) > 5:
            qa_history.pop(0)

        # Format previous Q&A neatly
        previous_qa = "\n\n---\n\n".join([f"**Q:** {q}\n**A:** {a}" for q, a in qa_history[:-1]])
        # Show last answer as new
        return (previous_qa + "\n\n---\n\n" if previous_qa else "") + f"**New:** {answer}"

    except Exception as e:
        return f"⚠ Error: {str(e)}"


def chatbot_response(message, history):
    return run_safe_query(message), history + [(message, run_safe_query(message))]

interface = gr.ChatInterface(
    fn=chatbot_response,
    title="Laptop Chatbot",
    description="Ask about laptops! I can recommend, compare, extract features, or summarize our conversation. (No phones!)",
    theme="soft",
    examples=[
        "Recommend laptops under ₹50,000",
        "Compare laptops with 16GB RAM",
        "Show laptops with i7 processor",
        "Summarize our conversation"
    ],
    cache_examples=False
)

if __name__ == "__main__":
    interface.launch()
