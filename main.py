from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever

# PDF path variable for api.py to import
PDF_PATH = "MLBook.pdf"

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
# print(pages[0].page_content)

# 2. Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=6000,
    chunk_overlap=200,
    add_start_index=True
)


# 3. Split the documents
chunks = text_splitter.split_documents(pages)



# 4. Verify the result
# print(f"You have {len(chunks)} chunks.")
# print(f"Content of first chunk: \n{chunks[0].page_content}")
#CREATE KEYWORD RETRIEVER
keyword_retriever = BM25Retriever.from_documents(chunks)
keyword_retriever.k = 7



#create embeddings
embedding_function = OllamaEmbeddings(
    model="qwen3-embedding:0.6b"
)



#create vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,
    persist_directory="./chroma_db_local"
)



#SETUP LLM
llm = OllamaLLM(
    model="llama3.2:latest",  # ← Change this to your model
    temperature=0.2,
    callbacks=[StreamingStdOutCallbackHandler()]
)



#create retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)


#MULTI QUERY RETRIEVER + ENSEMBLE RETRIEVER + COMPRESSION RETRIEVER
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever, # Your existing chroma retriever
    llm=llm
)
ensemble_retriever = EnsembleRetriever(
    retrievers=[multi_query_retriever, keyword_retriever],
    weights=[0.5, 0.5]
)
compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=ensemble_retriever
)



#PROMPT TEMPLATE
prompt_template = """You are a helpful AI assistant answering questions based on a machine learning textbook.
Use only the following context to answer the question. If the answer is not in the context, say "I don't have enough information in the provided context to answer this question."
Context:
{context}
Question: {question}
Answer: """
#create prompt
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)



# CREATE RAG CHAIN
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)




# INTERACTIVE QUESTION-ANSWERING LOOP
def ask_question(question):
    result = qa_chain.invoke({"query": question})
    print("Answer: ",result['result'])
    print(f"\n\n--- Sources ---")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"\nSource {i} (Page {doc.metadata.get('page', 'unknown')}):")
        print(doc.page_content[:200] + "...")
    return result




# Only run interactive loop when executed directly, not when imported
if __name__ == "__main__":
    while True:
        user_question = input("\nYour question: ").strip()
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
        if user_question:
            ask_question(user_question)