RAG Workflow 


1. Load the pdf , Using Library: langchain_community.document_loaders.PyPDFLoader
2. Split into chunks, Using Library: langchain_text_splitters.RecursiveCharacterTextSplitter
   Parameters:chunk_size=1000, chunk_overlap=200 
4. Create embeddings using qwen3-embedding model
5. Save all embeddings of chunks in Chroma vector database
6. Setup LLM 
   initially used smaller LLM gemma3:270m
   later used llama3.2 model to answer questions


7. Implemented Multi-Query Generation
   When you ask question, LLM creates 3 variations of  question
   
8. Semantic Search
   Converts each question type to vectors
   Searches chromadb for similar chunks 
   Returns 5 chunks per type of question

8. BM25 Keyword Search
   Searches for exact words like in the question
   Returns 5 chunks of same keywords

9. Ensemble Combining
   combines results from semantic and keyword searche
  

10. Rank the chunks by most relevent using langchain.retrievers.document_compressors.FlashrankRerank
    


11. create a prompt template , add context and query to it 
    

13. LLM reads the prompt and generates answer based on provided context

14. Return Results
Display the answer with source chunks and page numbers



