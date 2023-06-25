from langchain.prompts import PromptTemplate


def get_conversational_prompt():
  _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and recalls details from its past conversations. The AI offers support and is not judgemental. The human wants to create a website and the AI is focused on driving the conversation to understand what the human wants for the website and their overall goal. The AI is trying to have a long, deep conversation so they it can gather requirements for the development of the application.

Past conversation summaries:
{pinecone}

(You do not need to use these pieces of information if not relevant)

Current conversation summary:
{history}

Current conversation:
{chat_history_lines}

Human: {input}
AI:"""
  PROMPT = PromptTemplate(
    input_variables=["history", "input", "pinecone", "chat_history_lines"], template=_DEFAULT_TEMPLATE
    )
  return PROMPT


def get_end_prediction(human, output):
  prompt = f"""Return a Python True or False based on if you think a conversation has ended based on the final interaction.
 
  Human: {human}
  AI: {output}"""

  return prompt


def get_issue_planner_prompt(latest_conversation, related_conversations):
  newline = "\n"
  prompt = f"""
  Create a Python List of Github Issues from the lastest conversation.
  Each Issue should be concise and align with the user's vision for the application. Be critical and realistic.

  (You are not allowed to add packages. You can assume the environment vars are given.)

  The tech stack is:
  Database: Supabase
  Framework: Next.js
  Frontend: React.js
  CSS: Tailwind

  (You do not have to use these libraries if it is unnecessary)
  (You are not allowed to add packages. You can assume the environment vars are given.)


  Before finalizing the plan, search for human design patterns to consider integrating into your solution.

  Summaries of related conversations:
  {newline.join(related_conversations)}

  Summaries of current progress:

  Summary of latest conversation:
  {latest_conversation}
  """
  return prompt


def get_issue_executor_prompt(issue, type="feature"):
  if type == "feature":
    return f"""
  Create a Github Issue for this task:
  {issue}
  The Body should elaborate on the task in this template:
## 🚀 Feature
<CLEAR_CONCISE_DESCRIPTION>

## Motivation & Approach

<EXPLANATION>

<APPROACH>

  Name the Ticket appropriately.
  """


def get_hyde_prompt():
  prompt_template = """Generate code to resolve this Github Issue.

The tech stack is:
Database: Supabase
Framework: Next.js
Frontend: React.js
CSS: Tailwind

(You do not have to use these libraries if it is unnecessary)
(You are not allowed to add packages. You can assume the environment vars are given.)

Github Issue:
{issue}

Generate the response as a code file without any explanations.
"""
  prompt = PromptTemplate(input_variables=["issue"], template=prompt_template)
  return prompt


def get_code_planner_prompt(docs, issue):
  newline = "\n"
  prompt = f"""
Create an implementation for this Github Issue.

  The tech stack is:
  Database: Supabase
  Framework: Next.js
  Frontend: React.js
  CSS: Tailwind
(You do not have to use these libraries if it is unnecessary)
(You are not allowed to add packages. You can assume the environment vars are given.)

Issue:
{issue}

Relevant Documents:
{newline.join(docs)}

Let's think step by step.
"""

  return prompt


def get_code_executor_prompt(task):
  prompt = f"""
Solve the task.

For reference, the current tech stack is:
Database: Supabase
Framework: Next.js
Frontend: React.js
CSS: Tailwind

(You do not have to use these libraries if it is irrelevant)
(Do not add packages)

Create code for this task:
{task}
 (Do not worry about testing.)
"""
  return prompt