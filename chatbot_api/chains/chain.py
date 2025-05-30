"""This module contains the main application logic for the chatbot API.

It sets up the FastAPI application, configures middleware, and defines the API routes.
It also contains the logic for creating and configuring the language model and retriever.
"""

import os
import weaviate

from operator import itemgetter
from typing import Dict, List, Optional, Sequence
from chatbot_api.constants import WEAVIATE_DOCS_INDEX_NAME
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from chatbot_api.ingests import get_embeddings_model
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Weaviate
from langchain_ollama import ChatOllama
from langchain.docstore.document import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSequence,
    chain,
)
from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langsmith import Client
from chatbot_api.prompts import EVENTS_AGENT_RESPONSE_TEMPLATE

RESPONSE_TEMPLATE = EVENTS_AGENT_RESPONSE_TEMPLATE

COHERE_RESPONSE_TEMPLATE = EVENTS_AGENT_RESPONSE_TEMPLATE

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""


client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ChatRequest(BaseModel):
    """Model for the chat request.

    This model defines the structure of the chat request, which includes a question and an optional chat history.
    """
    question: str
    chat_history: Optional[List[Dict[str, str]]] = []
    session_id: Optional[str] = None  # Add this to track different chat sessions

    class Config:
        arbitrary_types_allowed = True


def get_retriever() -> BaseRetriever:
    """Create and return a retriever.

    This function creates a Weaviate retriever with the appropriate configuration and returns it.

    Returns:
        BaseRetriever: The created retriever.
    """
    weaviate_client = weaviate.Client(url="http://weaviate:8080")
    weaviate_client = Weaviate(
        client=weaviate_client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=get_embeddings_model(),
        by_text=False,
        attributes=["source", "title"],
    )
    return weaviate_client.as_retriever(search_kwargs=dict(k=6))


def create_retriever_chain(
    llm: LanguageModelLike, retriever: BaseRetriever
) -> Runnable:
    """Create and return a retriever chain.

    This function creates a retriever chain with the given language model and retriever, and returns it.

    Args:
        llm (LanguageModelLike): The language model to use in the chain.
        retriever (BaseRetriever): The retriever to use in the chain.

    Returns:
        Runnable: The created retriever chain.
    """
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def format_docs(docs: Sequence[Document]) -> str:
    """Format the given documents into a string.

    This function takes a sequence of documents and formats them into a string, where each document is enclosed in <doc> tags.

    Args:
        docs (Sequence[Document]): The documents to format.

    Returns:
        str: The formatted documents as a string.
    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    """Serialize the chat history in the given request.

    This function takes a chat request and serializes its chat history into a list of HumanMessage and AIMessage objects.

    Args:
        request (ChatRequest): The chat request containing the chat history to serialize.

    Returns:
        list: The serialized chat history.
    """
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    """Create and return a chain with the given language model and retriever.

    This function creates a chain with the given language model and retriever, and returns it.

    Args:
        llm (LanguageModelLike): The language model to use in the chain.
        retriever (BaseRetriever): The retriever to use in the chain.

    Returns:
        Runnable: The created chain.
    """
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")

    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
        .with_config(run_name="RetrieveDocs")
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    default_response_synthesizer = prompt | llm

    cohere_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COHERE_RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    @chain
    def cohere_response_synthesizer(input: dict) -> RunnableSequence:
        return cohere_prompt | llm.bind(source_documents=input["docs"])

    response_synthesizer = (
        default_response_synthesizer | StrOutputParser() 
    ).with_config(run_name="GenerateResponse")

    return (
        RunnablePassthrough.assign(chat_history=serialize_history)
        | context
        | response_synthesizer
    )


llama_local = ChatOllama(  
    model="llama3.2",  
    base_url="http://host.docker.internal:11434",  # This allows container to access host  
    temperature=0,  
)
cohere_command = None

llm = llama_local.configurable_alternatives(  
    ConfigurableField(id="llm"),  
    default_key="ollama_llama3_2",  

).with_fallbacks([llama_local])

retriever = get_retriever()
answer_chain = create_chain(llm, retriever)
