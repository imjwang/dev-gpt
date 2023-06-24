from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

class ConversationSummaryDB:
  def __init__(self, index, embedding):
    self.vectorstore = Pinecone(index, embedding, "text", namespace="conv")

  def save_summary(self, text):
    self.vectorstore.add_texts([text])

  def query(self, query: str):
    """Queries database for relevant past conversations"""
    docs = [doc.page_content for doc in self.vectorstore.similarity_search(query, k=4)]
    return docs
  

class RepositoryDB:
  def __init__(self, index, embedding):
    self.vectorstore = Pinecone(index, embedding, "text", namespace="repo")

  def load_documents(self, dir):
    loader = DirectoryLoader('./movie-picker', glob="**/*.js", loader_cls=TextLoader, use_multithreading=True)
    loaded_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_language(
      language=Language.JS,
      chunk_size = 8000,
      chunk_overlap  = 200,
      length_function = len
      )
    docs = text_splitter.split_documents(loaded_docs)
    self.vectorstore.add_documents(docs)


  def query(self, query: str):
    """Queries database for relevant past conversations"""
    docs = [doc.page_content for doc in self.vectorstore.similarity_search(query, k=4)]
    return docs