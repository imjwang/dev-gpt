import openai

from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
import os
from langchain.memory import CombinedMemory,ConversationSummaryMemory,ConversationBufferMemory
from agents.prompts.prompt import (get_conversational_prompt, get_end_prediction, 
                                   get_issue_planner_prompt, get_issue_executor_prompt, 
                                   get_code_planner_prompt, get_code_executor_prompt)
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage

from langchain.tools import StructuredTool
from langchain.chains import ConversationChain
from ast import literal_eval
from datetime import datetime

from ghapi.core import *
from ghapi.all import GhApi
from langchain.utilities import SerpAPIWrapper

from langchain.agents.agent_toolkits import FileManagementToolkit



import logging

def create_issue(title, body):
  """Creates a Github Issue with the following title and body to the current project"""
  github_token = os.environ['GITHUB_TOKEN']
  user = os.environ['GITHUB_USER']
  repo = os.environ['GITHUB_REPO']
  api = GhApi(owner=user, repo=repo, token=github_token)
  api.issues.create(title=title, body=body)

class IssueCreator:
  def __init__(self, conversation_summary_db, model="gpt-4-0613"):
    self.conversation_summary_db = conversation_summary_db

    try:
      openai.Model.retrieve(model)
      self.model = model
    except openai.InvalidRequestError as e:
      print(f"Failed to retrieve {model}, changing to gpt-3.5-turbo-0613")
      self.model = "gpt-3.5-turbo-0613"
    
    agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")]
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    create_issue_tool = StructuredTool.from_function(create_issue)
    executor_tools = [create_issue_tool]

    search = SerpAPIWrapper()
    planner_tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="helps you adopt human design patterns"
        ),
        Tool.from_function(
            name = "QueryConversationsDB",
            func=self.conversation_summary_db.query,
            description="useful to understand the user's intention, returns relevant timestamped conversations"
        ),
        StructuredTool.from_function(
            self.release_plan
        )
    ]

    llm = ChatOpenAI(temperature=0, model=self.model)
    self.planner = initialize_agent(
        planner_tools, 
        llm, 
        agent=AgentType.OPENAI_FUNCTIONS, 
        agent_kwargs=agent_kwargs, 
        memory=memory
    )

    self.executor = initialize_agent(
        executor_tools, 
        llm, 
        agent=AgentType.OPENAI_FUNCTIONS, 
    )
    
    self.plans = []

  def run(self, latest_conversation):
    docs = self.conversation_summary_db.query(latest_conversation)
    prompt = get_issue_planner_prompt(latest_conversation, docs)
    res = self.planner.run(prompt)
    return res
  
  def release_plan(self, plans:str):
    """Creates Github Issues from a Python String representation of a List[String]"""
    try:
      self.plans = literal_eval(plans)
    except:
      self.plans = []
    for plan in self.plans:
      prompt = get_issue_executor_prompt(plan)
      self.executor.run(prompt)
       
      
class ConversationalAgent:
  def __init__(self, conversation_memory, model="gpt-4-0613"):
      try:
        openai.Model.retrieve(model)
        self.model = model
        
      except openai.InvalidRequestError as e:
        print(f"Failed to retrieve {model}, changing to gpt-3.5-turbo-0613")
        self.model = "gpt-3.5-turbo-0613"

      llm = ChatOpenAI(temperature=0, model=model)

      self.summary_memory = ConversationSummaryMemory(llm=llm, input_key="input")
      current_chat_memory = ConversationBufferMemory(memory_key="chat_history_lines",
                                              input_key="input")
      
      memory = CombinedMemory(memories=[conversation_memory, self.summary_memory, current_chat_memory])

      prompt = get_conversational_prompt()
      # TODO ADD TOOLS

      self.agent = ConversationChain(
        llm=llm, 
        prompt=prompt,
        memory=memory
        )
  

  def run(self, prompt):
    # time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    result = self.agent.predict(input=prompt)
    return result
  

  def get_summary(self):
    return f'{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")} {self.summary_memory.buffer}' 


class Verifier:
  def __init__(self, model="gpt-4-0613"):
    try:
      openai.Model.retrieve(model)
      self.model = model
    
    except openai.InvalidRequestError as e:
      print(f"Failed to retrieve {model}, changing to gpt-3.5-turbo-0613")
      self.model = "gpt-3.5-turbo-0613"

    self.llm = ChatOpenAI(temperature=0, model=model)

  def predict(self, human, model):
    res = get_end_prediction(human, model)

    conversation_over = self.llm.predict(res)
    try:
      rtn = literal_eval(conversation_over)
    except:
      rtn = False

    return rtn


class Coder:
  def __init__(self, repo_db, model="gpt-4-0613"):
    try:
      openai.Model.retrieve(model)
      self.model = model
    
    except openai.InvalidRequestError as e:
      print(f"Failed to retrieve {model}, changing to gpt-3.5-turbo-0613")
      self.model = "gpt-3.5-turbo-0613"

    self.llm = ChatOpenAI(temperature=0, model=model)
    self.repo_db = repo_db

    tools = FileManagementToolkit(
      root_dir='./movie-picker/src',
      selected_tools=["read_file", "write_file"],
    ).get_tools()

    read_tool, write_tool = tools

    read_tool.name = "Read"
    read_tool.description = "useful to understand contents of a file"

    write_tool.name = "Message"
    write_tool.description = "useful to send code as a message for user"

    test_tools = [read_tool, write_tool]
  

    planner_agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": SystemMessage(content="You are an expert software design AI. Produce concise plans with clear instructions.")
    }

    executor_agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": SystemMessage(content="You are an expert software development AI. Please help the human by saving code output using functions provided. The functions require valid JSON.")
    }
    
    planner_memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
    executor_memory = ConversationBufferMemory(memory_key="memory", return_messages=True)


    self.planner = initialize_agent(
      test_tools, 
      self.llm, 
      agent=AgentType.OPENAI_FUNCTIONS, 
      agent_kwargs=planner_agent_kwargs, 
      memory=planner_memory
    )

    self.executor = initialize_agent(
      test_tools,
      self.llm,
      agent=AgentType.OPENAI_FUNCTIONS,
      agent_kwargs=executor_agent_kwargs,
      memory=executor_memory
    )


  def plan_and_execute(self, issue):
    task = self.plan(issue)
    res = self.execute(task)
    return res


  def plan(self, issue):
    docs = self.repo_db.query(issue)
    prompt = get_code_planner_prompt(docs, issue)
    task = self.planner.run(prompt)
    return task
  
  
  def execute(self, task):
    prompt = get_code_executor_prompt(task)
    res = self.executor.run(prompt)
    return res
  
  
  def reflect(self, response):
    prompt = f"Given the AI Response, predict if the AI has accomplished the task of writing code to file. Return a Python True or False. Response: {response}"
    res = self.llm.predict(prompt)
    try:
      rtn = literal_eval(res)
    except:
      rtn = False
    return rtn