# app.py
"""
Movie Recommendation System (Cleaned from Colab)
No ADK. No fake packages. Just working RAG.
"""

import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.chains import RetrievalQA
from langchain.schema import Document


# === 1. Load and Clean Data ===
def load_and_clean_data():
    data_path = "data/IMDB top 1000.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    movies = pd.read_csv(data_path)
    movies.columns = movies.columns.str.strip()  # Clean column names

    # Clean missing values
    movies = movies.dropna(subset=['Title', 'Description'])
    movies['Genre'] = movies['Genre'].fillna('Unknown')
    movies['Cast'] = movies['Cast'].fillna('Unknown')
    movies['Rate'] = pd.to_numeric(movies['Rate'], errors='coerce')
    movies['Rate'] = movies['Rate'].fillna(movies['Rate'].mean())

    # Remove duplicates and strip whitespace
    movies = movies.drop_duplicates()
    for col in ['Title', 'Genre', 'Description', 'Cast']:
        if col in movies.columns:
            movies[col] = movies[col].str.strip()

    print(f"✅ Loaded {len(movies)} movies")
    return movies


# === 2. Create MetaText ===
def create_meta_text(row):
    return (
        f"Title: {row['Title']}. "
        f"Genre: {row['Genre']}. "
        f"Plot: {row['Description']}. "
        f"Stars: {row['Cast']}. "
        f"Rating: {row['Rate']}"
    )


# === 3. Load Chroma Vector Store ===
def load_vector_store():
    persist_dir = "models/chroma_db"
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"Chroma DB not found at {persist_dir}. Did you copy it from Colab?")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="movie_chunks"
    )
    print("✅ Chroma vector store loaded")
    return vector_store


# === 4. Setup LLM (GPT-2) ===
def get_llm():
    print("Loading GPT-2... (this may take a moment)")
    llm_pipeline = pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=256,
        temperature=0.4,
        pad_token_id=50256,
    )
    return HuggingFacePipeline(pipeline=llm_pipeline)


# === 5. Setup Prompt Template ===
def get_prompt():
    template = """
You are a movie recommendation assistant.

Analyze the following movie descriptions provided as context:
{context}

Based on the user's request: "{input}", provide a clear, concise, and well-structured recommendation explanation.

Your answer should be in natural, human-readable form and focus on the following aspects:

1.  **Genre Alignment:** Explain how the recommended movies align with the genres mentioned in the user's request.
2.  **Star Relevance:** Highlight any relevant stars from the recommended movies that might appeal to the user based on their request.
3.  **Plot Similarity:** Describe how the plots of the recommended movies are similar to what the user is looking for.

Structure your answer clearly using bullet points or numbered lists for each of the three aspects.

Do NOT repeat the context or the user's request in your answer. Only provide the final recommendation explanation.

Answer:
"""
    return PromptTemplate.from_template(template)


# === 6. Main App ===
if __name__ == "__main__":
    # Load data (for metadata only — Chroma already has embeddings)
    movies = load_and_clean_data()
    movies['MetaText'] = movies.apply(create_meta_text, axis=1)

    # Load vector store
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Setup LLM and prompt
    llm = get_llm()
    prompt = get_prompt()

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("\n🎬 Movie Recommender Ready!")
    print("Type 'quit' to exit.\n")

    # Interactive loop
    while True:
        query = input("Enter your movie preference: ").strip()
        if query.lower() == 'quit':
            break
        if not query:
            continue

        try:
            result = qa_chain.invoke({"query": query})
            print("\n📝 Recommendation:")
            print(result["result"])
            print("\n" + "-" * 50 + "\n")
        except Exception as e:
            print(f"Error: {e}")