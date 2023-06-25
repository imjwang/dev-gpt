import os
from langchain.embeddings.openai import OpenAIEmbeddings
from agents.agent import ConversationalAgent, Verifier, IssueCreator, Coder
from ghapi.core import *
from ghapi.all import GhApi
import subprocess
import random
from db.conversations import RepositoryDB, ConversationSummaryDB, init_pinecone_index
import argparse


def generate_random_number():
  return random.randint(1, 100000)


def get_issue(num):
  repo = os.environ['GITHUB_REPO']
  github_token = os.environ['GITHUB_TOKEN']
  user = os.environ['GITHUB_USER']
  gh = GhApi(owner=user, repo=repo, token=github_token)

  issue = gh.issues.get(issue_number=num)
  return issue.title, issue.body


def git_checkout_add_commit_push(branch_name="testing123"):
  repo = os.environ['GITHUB_REPO']

  os.chdir(f"./{repo}")

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
  parser = argparse.ArgumentParser(description='Fetch GitHub issue')

  parser.add_argument('--issue', type=int, help='GitHub issue number')

  args = parser.parse_args()

  embed = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_key=os.environ["OPENAI_API_KEY"]
  )

  index = init_pinecone_index(index_name=INDEX_NAME)


  conversation_db = ConversationSummaryDB(index, embed)
  repo_db = RepositoryDB(index, embed)
  repo = os.environ['GITHUB_REPO']
  repo_db.load_documents(f"./{repo}")

  retriever = conversation_db.get_retriever_memory()

  title, body = get_issue(args.issue)

  issue = f"""
  Title: {title}
  Body:
  {body}
  """


  Chatter = ConversationalAgent(retriever)
  Stopper = Verifier()
  Issuer = IssueCreator(conversation_db)

  print("Note: if you want to end conversation, say goodbye.")

  name = input("Please enter your name: ")

  while True:
    human_input = input(f"{name}: ")
    model_output = Chatter.run(human_input)
    print(f"AI MODEL OUTPUT: {model_output}\n")

    over = Stopper.predict(human_input, model_output)
    if over:
      print("CONVERSATION HAS ENDED")
      conversation_summary = Chatter.get_summary()
      print("SAVING SUMMARY TO DB")
      conversation_db.save_summary(conversation_summary)
      break

  print("CREATING GITHUB ISSUES")
  Issuer.run(latest_conversation=conversation_summary)

  print("CODING")
  coder = Coder(repo_db)
  while True:
    res = coder.plan_and_execute(issue)
    print(res)
    flag = coder.reflect(res)
    if flag:
       break
    issue = res

  print("CREATING PULL REQUEST")
  pull_request_number = git_checkout_add_commit_push(f"dev-gpt-{generate_random_number()}")
  print(f"Pull request created with number {pull_request_number}")



if __name__ == "__main__":
    main()