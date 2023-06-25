import logging
import click
import openai
import os
from ghapi.core import *
from ghapi.all import GhApi
from langchain.embeddings.openai import OpenAIEmbeddings
from agents.agent import Coder
import subprocess



from db.conversations import RepositoryDB, init_pinecone_index
 

def git_checkout_add_commit_push(branch_name="testing123"):

  os.chdir('movie-picker')

  # Git checkout -b test-branch
  checkout_cmd = f'git checkout -b {branch_name}'
  subprocess.call(checkout_cmd, shell=True)
  # Git add .
  add_cmd = 'git add .'
  subprocess.call(add_cmd, shell=True)
  # Git commit -m "asdf"
  commit_cmd = f'git commit -m "commit by gpt-dev"'
  subprocess.call(commit_cmd, shell=True)
  push_cmd = f'git push --set-upstream origin {branch_name}'
  subprocess.call(push_cmd, shell=True)
  # Create an instance of the GhApi client
  github_token = os.environ['GITHUB_TOKEN']
  user = os.environ['GITHUB_USER']
  repo = os.environ['GITHUB_REPO']
  gh = GhApi(owner=user, repo=repo, token=github_token)
  # Create the pull request
  pr = gh.pulls.create(base="main", head=branch_name, title="test", body="test")
  # Return the pull request number
  return pr.number

 
INDEX_NAME ='pinecone-hackathon'
EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'


def main():
  embed = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_key=os.environ["OPENAI_API_KEY"]
  )

  index = init_pinecone_index(index_name=INDEX_NAME)
  repo_db = RepositoryDB(index, embed)
  repo_db.load_documents('./movie-picker')


  issue = """
  Title: Issue #2: Implement User Feedback Form
  Body:
  🚀 Feature
Create a feedback form for users to like or dislike the project. This form should be the main feature of the dashboard.

Motivation & Approach
To understand user preferences and improve the project based on user feedback, we need to implement a feedback form. This form will allow users to express their likes or dislikes about the project.

The approach will be to design and implement this form as a main feature of the dashboard. It should be user-friendly and easily accessible.
  """

  # coder = Coder(repo_db)
  # while True:
  #   res = coder.plan_and_execute(issue)
  #   print(res)
  #   flag = coder.reflect(res)
  #   if flag:
  #      break
  #   issue = res

  pull_request_number = git_checkout_add_commit_push("testing")
  print(f"Pull request created with number {pull_request_number}")



if __name__ == "__main__":
    main()