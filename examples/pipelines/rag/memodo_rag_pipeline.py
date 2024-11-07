"""
title: Memodo RAG Pipeline
author: Memodo GmbH
date: 2024-11-06
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the LangChain library.
requirements: langchain
"""

import os
import chromadb
import logging

from dotenv import load_dotenv
from typing import Generator, Iterator, List, Union
from pydantic import BaseModel, SecretStr
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from operator import itemgetter

class Pipeline:
    class Valves(BaseModel):
        VECTOR_DB_HOST: str
        VECTOR_DB_PORT: str
        COLLECTION_NAME: str
        MODEL_NAME: str

    class Retreive:
        def __init__(self, model, host, port, collection_name, openai_api_key):
            self.printWithEmphasis(f"Using model: {model}")

            if model.startswith("gpt"):
                self.model_instance = ChatOpenAI(api_key=SecretStr(openai_api_key), model=model)
                embeddings = OpenAIEmbeddings()
            else:
                self.model_instance = Ollama(model=model)
                embeddings = OllamaEmbeddings(model=model)

            client = chromadb.HttpClient(host, port)
            self.vector_db_documents = client.get_collection(
                    name=collection_name
                )

            logging.basicConfig()
            logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
            
        def printWithEmphasis(toBePrinted):
            print("**********************************************")
            print(f"{toBePrinted}")
            print("**********************************************")
        
        def lineListOutputParser(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            self.printWithEmphasis(f"Multi-query questions: {lines}")
            return lines  

        def query_db(self, questions):
            results = self.vector_db_documents.query(
                query_texts=questions,
                n_results=10
            )
            self.printWithEmphasis(f"Raw Results: {results}")
            self.printWithEmphasis(f"Result Documents: {results['documents']}")
            return results["documents"]

        def preview_results(self, results):
            self.printWithEmphasis(f"Preview Results: {results}")
            return results
        
        def do_rag(self, user_question):
            multiquery_template = """
            You are an AI language model assistant.

            Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database.

            By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search.

            Provide these alternative questions separated by newlines.

            Original question: {question}
            """
            
            rag_template = """
            Answer the question based on the context below. If you can't answer the question, say "I don't know." Do not make up new information.

            Context: {context}

            Question: {question}
            """

            multiquery_prompt = PromptTemplate.from_template(multiquery_template)
            rag_prompt = PromptTemplate.from_template(rag_template)

            parser = StrOutputParser()

            chain = (
                {
                    "question": lambda x: itemgetter(x["question"])
                }
                | multiquery_prompt
                | self.model_instance
                | self.lineListOutputParser
                | {
                    "context": lambda x: self.query_db(x),
                    "question": lambda _: user_question
                }
                | rag_prompt
                | self.model_instance
                | {
                    "content": lambda x: self.preview_results(x)
                }
                | parser
            )

            try:
                self.printWithEmphasis(f"User's question: {user_question}")
                result = chain.invoke({"question": user_question})
                self.printWithEmphasis(f"Result: {result}")
            except Exception as e:
                self.printWithEmphasis(e)

    def __init__(self):
        self.retreive = None
        self.name = "Memodo RAG Pipeline"
        self.valves = self.Valves(
            **{
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
                "VECTOR_DB_HOST": os.getenv("VECTOR_DB_HOST", "localhost"),
                "VECTOR_DB_PORT": os.getenv("VECTOR_DB_PORT", "8000"),
                "COLLECTION_NAME": os.getenv("COLLECTION_NAME", "test"),
                "MODEL_NAME": os.getenv("MODEL_NAME", "llama3.2:3b")
            }
        )

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        self.retreive = self.Retreive(self.valves.MODEL_NAME, self.valves.VECTOR_DB_HOST, self.valves.VECTOR_DB_PORT, self.valves.COLLECTION_NAME, self.valves.OPENAI_API_KEY)
        

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")

        try:
            return self.retreive.do_rag(user_message)
        except Exception as e:
            return f"Error: {e}"