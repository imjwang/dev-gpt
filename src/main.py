import logging
import click
import openai
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from agents.agent import ConversationalAgent, Verifier, IssueCreator


from db.conversations import RepositoryDB, ConversationSummaryDB, init_pinecone_index
 
 
INDEX_NAME ='pinecone-hackathon'
EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'


def main():

  embed = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_key=os.environ["OPENAI_API_KEY"]
  )

  index = init_pinecone_index(index_name=INDEX_NAME)


  conversation_db = ConversationSummaryDB(index, embed)
  repo_db = RepositoryDB(index, embed)
  repo_db.load_documents('./movie-picker')

  retriever = conversation_db.get_retriever_memory()

  Chatter = ConversationalAgent(retriever)
  Stopper = Verifier()
  Issuer = IssueCreator(conversation_db)

  while True:
    human_input = input("conversation: ")
    model_output = Chatter.run(human_input)
    print(model_output)

    over = Stopper.predict(human_input, model_output)
    if over:
      conversation_summary = Chatter.get_summary()
      conversation_db.save_summary(conversation_summary)
      break

  Issuer.run(latest_conversation=conversation_summary)




if __name__ == "__main__":
    main()