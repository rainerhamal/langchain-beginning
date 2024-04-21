import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

# !LangSMith sample on how to run it using @traceable decorator
# # Auto-trace LLM calls in-context
# client = wrap_openai(openai.Client())

# @traceable
# def pipeline(user_input:str):
#     result = client.chat.completions.create(
#         messages=[{"role": "user", "content": user_input}],
#         model="gpt-3.5-turbo"
#     )
#     return result.choices[0].message.content 
# pipeline("Hello, world!")
# # Out: Hello there! How can I asssit you today?

# !Initialise the model
# llm = OpenAI() #take a string prompt as input and output a string completion
llm = ChatOpenAI(model="gpt-3.5-turbo-0125") #take a list of chat messages as input and they return an AI message as output

# ?try using the model by invoking it
# llm.invoke("how can langsmith help with testing?")

# !Guide LLM response with a prompt template.Prompt templates convert raw user input to better input to the LLM.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a world class technical documentation writer."),
    ("user", "{input}")
])
# ?combine these into a simple llm chain
# chain = prompt | llm
# ?invoke the chain and ask question. it should respond in a more proper tone for a technical writer
# chain.invoke({"input": "how can langsmith help with testing?"})

# ! add a simple output parser to convert the chat message to a string
output_parser = StrOutputParser()
# ?add this to the previous chain:
chain = prompt | llm | output_parser
chain.invoke({"input": "how can langsmith help with testing?"})