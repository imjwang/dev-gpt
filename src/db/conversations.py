import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.chat_models import ChatOpenAI
import pinecone
from langchain.vectorstores import Pinecone
import logging
from langchain.memory import VectorStoreRetrieverMemory, ReadOnlySharedMemory
from agents.prompts.prompt import get_hyde_prompt


def init_pinecone_index(index_name='pinecone-hackathon'):
  
  pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENV"]
  )

  # pinecone.create_index(
  #   name=index_name,
  #   metric='cosine',
  #   dimension=1536  # 1536 dim of text-embedding-ada-002
  # )

  index = pinecone.Index(index_name)

  return index

class ConversationSummaryDB:
  def __init__(self, index, embedding):
    self.vectorstore = Pinecone(index, embedding.embed_query, "text", namespace="conv")

  def save_summary(self, text):
    self.vectorstore.add_texts([text])

  def query(self, query: str):
    """Queries database for relevant past conversations"""
    docs = [doc.page_content for doc in self.vectorstore.similarity_search(query, k=10)]
    return docs
  
  def get_retriever_memory(self, read_only=True):
    retriever = self.vectorstore.as_retriever(search_kwargs=dict(k=10))
    pinecone_memory = VectorStoreRetrieverMemory(memory_key="pinecone", retriever=retriever)
    if read_only:
      rpine_cone_memory = ReadOnlySharedMemory(memory=pinecone_memory)
      return rpine_cone_memory

    return read_only
  

class RepositoryDB:
  def __init__(self, index, embed):
    self.index = index
    llm = ChatOpenAI(temperature=0, model="gpt-4-0613")

    prompt =  get_hyde_prompt()

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    hyde_embedding = HypotheticalDocumentEmbedder(
      llm_chain=llm_chain, base_embeddings=embed
    )
    self.hyde_vectorstore = Pinecone(index, hyde_embedding.embed_query, "text", namespace="repo")
    self.vectorstore = Pinecone(index, embed.embed_query, "text", namespace="repo")

  def load_documents(self, dir):
    # fresh files
    self.index.delete(delete_all=True, namespace="repo")

    loader = DirectoryLoader(dir, glob="**/*.js", loader_cls=TextLoader, use_multithreading=True)
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
    docs = [f"Score: {score} File Path: {doc.metadata['source']} File Contents:{doc.page_content}" for doc, score in self.hyde_vectorstore.similarity_search_with_score(query, k=4)]
    return docs
  