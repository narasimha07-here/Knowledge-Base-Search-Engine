import os
import requests
import streamlit as st
from typing import Dict, Any, List, Optional

API_BASE_URL = st.secrets.get("API_BASE_URL")

class APIError(Exception):
    def __init__(self, message, status_code=None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class APIClient:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        
    def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/register",
            json={"username": username, "email": email, "password": password}
        )
        response.raise_for_status()
        return response.json()
    
    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/login",
            data={"username": username, "password": password}
        )
        response.raise_for_status()
        return response.json()
    
    def upload_files(self, userid: int, files: List[Any]) -> Dict[str, Any]:
        files_data = [("files", (f.name, f, f.type)) for f in files]
        data = {"userid": userid}
        response = requests.post(
            f"{self.base_url}/upload",
            data=data,
            files=files_data
        )
        response.raise_for_status()
        return response.json()
    
    def get_documents(self, userid: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        response = requests.get(
            f"{self.base_url}/documents",
            params={"userid": userid, "limit": limit, "offset": offset}
        )
        response.raise_for_status()
        return response.json()
    
    def delete_document(self, docid: int, userid: int) -> Dict[str, Any]:
        response = requests.delete(
            f"{self.base_url}/documents/{docid}",
            params={"userid": userid}
        )
        response.raise_for_status()
        return response.json()
    
    def semantic_search(self, query: str, userid: int, topk: int = 10) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/search/semantic",
            json={"query": query, "userid": userid, "topk": topk}
        )
        response.raise_for_status()
        return response.json()
    
    def hybrid_search(
        self,
        query: str,
        userid: int,
        topk: Optional[int] = None,
        keyword_weight: float = 0.3,
        semantic_weight: float = 0.7,
        use_all_chunks: bool = True
    ) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/search/hybrid",
            json={
                "query": query,
                "userid": userid,
                "topk": topk,
                "keyword_weight": keyword_weight,
                "semantic_weight": semantic_weight,
                "use_all_chunks": use_all_chunks
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_search_history(self, userid: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        response = requests.get(
            f"{self.base_url}/search/history",
            params={"userid": userid, "limit": limit, "offset": offset}
        )
        response.raise_for_status()
        return response.json()
    
    def get_user_stats(self, userid: int) -> Dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/stats",
            params={"userid": userid}
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()