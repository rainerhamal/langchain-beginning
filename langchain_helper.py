import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

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
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a world class technical documentation writer."),
#     ("user", "{input}")
# ])
# ?combine these into a simple llm chain
# chain = prompt | llm
# ?invoke the chain and ask question. it should respond in a more proper tone for a technical writer
# chain.invoke({"input": "how can langsmith help with testing?"})

# ! add a simple output parser to convert the chat message to a string
# output_parser = StrOutputParser()
# # ?add this to the previous chain:
# chain = prompt | llm | output_parser
# chain.invoke({"input": "how can langsmith help with testing?"})

# !Retrieval Chain
# ?import and use WebBaseLoader
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
# ?index it into a vectorstore. This requires a few components, namely an embedding model and a vectorstore.
embeddings = OpenAIEmbeddings()
# ?We will use a simple local vectorstore, FAISS, for simplicity's sake.
# ?building our index
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# ?Now that we have this data indexed in a vectorstore, we will create a retrieval chain. This chain will take an incoming question, look up relevant documents, then pass those documents along with the original question into an LLM and ask it to answer the original question.
prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:
    <context>
        {context}
    </context>
    
    Question: {input}"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
# ?we want the documents to first come from the retriever we just set up.use the retriever to dynamically select the most relevant documents and pass those in for a given question.
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
# ?Invoke this chain. This returns a dictionary - the response from the LLM is in the answer key
response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
print(response["answer"])
# LangSmith offers several features that can help with testing:...
# !This answer should be much more accurate! ^
