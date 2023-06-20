import openai
from ghapi.core import *
from ghapi.all import GhApi
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
import os

from langchain.tools import StructuredTool

import logging

def create_issue(title, body):
  """Creates a Github Issue with the following title and body to the current project"""
  github_token = os.environ['GITHUB_TOKEN']
  api = GhApi(owner='imjwang', repo='climate', token=github_token)
  api.issues.create(title=title, body=body)

class IssueCreator:
    def __init__(self, model="gpt-4-0613"):
      try:
        openai.Model.retrieve(model)
        self.model = model
      except openai.InvalidRequestError as e:
        print(f"Failed to retrieve {model}, changing to gpt-3.5-turbo-0613")
        self.model = "gpt-3.5-turbo-0613"
      
      openai_api_key = os.environ['OPENAI_API_KEY']
      llm = ChatOpenAI(temperature=0, model=model, openai_api_key=openai_api_key)
      tool = StructuredTool.from_function(create_issue)
      tools = [tool]
      self.agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)


    def run(self, prompt):
      self.agent.run(prompt)
         
      

