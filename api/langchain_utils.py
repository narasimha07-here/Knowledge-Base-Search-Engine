import os
import logging
from typing import List,Dict,Any,Callable,Optional
from pathlib import Path
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("llm")

class LLMManager:
    def __init__(self):
        model = os.getenv("model")
        api_key = os.getenv("OPENAI_APIKEY")
        api_base = os.getenv("OPENAI_APIBASE", "https://api.openai.com/v1")
        self.llm = ChatOpenAI(model=model, openai_api_key=api_key, openai_api_base=api_base)

        self.supported_types = [".txt", ".md", ".pdf", ".csv", ".json"]
        self.prompt = ChatPromptTemplate.from_template(
            "You are an AI assistant that answers questions based on provided context.\n"
            "Use the context to answer the question.\n"
            "If unsure, say so.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        self.prompt1 = ChatPromptTemplate.from_template(
            "You are an AI assistant that answers questions based on provided context.\n"
            "Use the context to answer the question in depth even for single line.\n"
            "If unsure, say so.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        logger.info("LLM initialized")

    async def load_file(self, filepath: str) -> List[Document]:
        path = Path(filepath)
        suffix = path.suffix.lower()
        if suffix == ".txt" or suffix == ".md":
            text = path.read_text(encoding="utf-8", errors="ignore")
            return [Document(page_content=text, metadata={"source": str(path)})]
        elif suffix == ".csv":
            text = path.read_text(encoding="utf-8", errors="ignore")
            return [Document(page_content=text, metadata={"source": str(path)})]
        elif suffix == ".json":
            text = path.read_text(encoding="utf-8", errors="ignore")
            return [Document(page_content=text, metadata={"source": str(path)})]
        elif suffix == ".pdf":
            try:
                reader = PdfReader(str(path))
                content = "\n".join(page.extract_text() or "" for page in reader.pages)
                return [Document(page_content=content, metadata={"source": str(path)})]
            except Exception as e:
                raise RuntimeError(f"PDF parse error: {e}")
        else:
            raise RuntimeError(f"Unsupported extension: {suffix}")

    def is_supported_file(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.supported_types

    async def generate_response(self, retriever, question: str) -> Dict[str, Any]:
        try:
            # Retrieve docs
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([d.page_content for d in docs])
            chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.prompt1
                | self.llm
                | StrOutputParser()
            )
            answer = chain.invoke({"context": context, "question": question})
            sources = [{"source": d.metadata.get("source", "unknown")} for d in docs]
            return {
                "response": answer,
                "sources": sources,
                "retrieved_chunks": len(docs),
                "success": True
            }
        except Exception as e:
            logger.error(f"LLM generate_response error: {e}")
            return {"response": "", "sources": [], "retrieved_chunks": 0, "success": False}

    async def generate_hybrid_response(self, retriever, question: str, hybrid_docs_func: Optional[Callable[[str], List[Document]]] = None) -> Dict[str, Any]:
        try:
            docs = hybrid_docs_func(question) if hybrid_docs_func else retriever.get_relevant_documents(question)
            context = "\n\n".join([d.page_content for d in docs])
            chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            answer = chain.invoke({"context": context, "question": question})
            sources = [{"source": d.metadata.get("source", "unknown")} for d in docs]
            return {
                "response": answer,
                "sources": sources,
                "retrieved_chunks": len(docs),
                "success": True
            }
        except Exception as e:
            logger.error(f"LLM generate_hybrid_response error: {e}")
            return {"response": "", "sources": [], "retrieved_chunks": 0, "success": False}
