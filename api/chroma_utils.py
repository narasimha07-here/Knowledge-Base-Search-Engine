import os
import uuid
import logging
from typing import List,Dict,Any,Optional
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
import re
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("vec")

class VectorStore:
    def __init__(self, persist_dir: str, model: str):
        self.persist_dir = Path(os.getenv("VECTORDIR"))
        self.persist_dir.mkdir(exist_ok=True, parents=True)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": os.getenv("HF_DEVICE", "cpu")},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        self.splitter = SemanticChunker(embeddings=self.embeddings, breakpoint_threshold_type="standard_deviation")
        self._collections: Dict[str, Chroma] = {}
        
        logger.info(f"VectorStore initialized at {self.persist_dir}")

    def user_collection_name(self, uid: int) -> str:
        return f"user_{uid}_docs"

    def _get_user_persist_dir(self, collection_name: str) -> Path:
        """Get the persist directory for a specific user collection"""
        user_dir = self.persist_dir / collection_name
        user_dir.mkdir(exist_ok=True, parents=True)
        return user_dir

    def get_collection(self, name: str, docs: Optional[List[Document]] = None) -> Chroma:
        """Get or create a collection with proper per-user isolation"""
        try:

            if name in self._collections:
                logger.info(f"Returning cached collection {name}")
                return self._collections[name]

            user_persist_dir = self._get_user_persist_dir(name)
            
            # if collection already exists 
            collection_exists = (user_persist_dir.exists() and 
                               any(user_persist_dir.iterdir()) and
                               (user_persist_dir / "chroma.sqlite3").exists())
            
            if collection_exists:
                logger.info(f"Loading existing collection {name} from {user_persist_dir}")
                vs = Chroma(
                    collection_name=name, 
                    embedding_function=self.embeddings, 
                    persist_directory=str(user_persist_dir)
                )
            else:
                logger.info(f"Creating new collection {name} at {user_persist_dir}")
                if docs:
                    vs = Chroma.from_documents(
                        collection_name=name, 
                        documents=docs, 
                        embedding=self.embeddings, 
                        persist_directory=str(user_persist_dir)
                    )
                else:
                    vs = Chroma(
                        collection_name=name, 
                        embedding_function=self.embeddings, 
                        persist_directory=str(user_persist_dir)
                    )
            
            # Cache the collection
            self._collections[name] = vs
            logger.info(f"Collection {name} ready with persist_dir: {user_persist_dir}")
            return vs
            
        except Exception as e:
            logger.error(f"Error with collection {name}: {e}")
            raise

    def add_docs(self, collection_name: str, docs: List[Document], filehash: str) -> Dict[str, Any]:
        try:
            logger.info(f"Splitting {len(docs)} documents for collection {collection_name}")
            chunks = self.splitter.split_documents(docs)
            
            for chunk in chunks:
                chunk.metadata["filehash"] = filehash
                chunk.metadata["docid"] = chunk.metadata.get("docid", str(uuid.uuid4()))
                chunk.metadata["collection"] = collection_name
            
            vs = self.get_collection(collection_name)
            vs.add_documents(chunks)

            data = vs.get()
            total_docs = len(data.get("ids", [])) if data else 0
            
            logger.info(f"Added {len(chunks)} chunks to {collection_name}. Total docs in collection: {total_docs}")
            
            return {
                "success": True, 
                "chunks_added": len(chunks), 
                "original_docs": len(docs), 
                "collection": collection_name,
                "total_in_collection": total_docs
            }
            
        except Exception as e:
            logger.error(f"Error adding docs to {collection_name}: {e}")
            return {
                "success": False, 
                "error": str(e), 
                "chunks_added": 0, 
                "original_docs": len(docs)
            }

    def keyword_search(self, collection_name: str, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            vs = self.get_collection(collection_name)
            data = vs.get()
            
            if not data or not data.get("documents"):
                logger.warning(f"No documents found in collection {collection_name}")
                return []
            
            total_docs = len(data.get('documents', []))

            if k is None:
                k = total_docs
                logger.info(f"Keyword search using ALL {k} chunks from {collection_name}")
            
            query_tokens = self._tokenize(query.lower())
            scored = []
            
            for doc_text, metadata, doc_id in zip(data["documents"], data["metadatas"], data["ids"]):
                score = self._bm25_score(query_tokens, doc_text.lower())
                if score > 0:
                    scored.append({
                        "content": doc_text, 
                        "metadata": metadata, 
                        "keyword_score": score, 
                        "docid": doc_id
                    })
            
            scored.sort(key=lambda x: x["keyword_score"], reverse=True)
            return scored  # Return ALL scored results, no truncation
            
        except Exception as e:
            logger.error(f"Keyword search error in {collection_name}: {e}")
            return []

    def hybrid_search(self, collection_name: str, query: str, k: Optional[int] = None, 
                     keyword_weight: float = 0.3, semantic_weight: float = 0.7) -> Dict[str, Any]:
        try:
            vs = self.get_collection(collection_name)
            all_data = vs.get()
            
            total_chunks = len(all_data.get("ids", [])) if all_data else 0
            logger.info(f"Hybrid search in {collection_name}: found {total_chunks} total chunks")
            
            if total_chunks == 0:
                logger.warning(f"No documents in collection {collection_name}")
                return {
                    "success": False, 
                    "error": "No documents in collection", 
                    "results": [], 
                    "total": 0, 
                    "query": query,
                    "collection_name": collection_name,
                    "total_chunks": 0
                }

            k_actual = total_chunks  
            logger.info(f"HYBRID SEARCH USE ALL {k_actual} CHUNKS")

            semantic_results = vs.similarity_search_with_score(query, k=k_actual)
            semantic_dict: Dict[str, Dict[str, Any]] = {}
            
            for doc, distance in semantic_results:
                docid = doc.metadata.get("docid", str(uuid.uuid4()))
                semantic_dict[docid] = {
                    "doc": doc,
                    "semantic_score": float(1.0 - distance),  
                }

            keyword_results = self.keyword_search(collection_name, query, k=None) 
            keyword_dict: Dict[str, Dict[str, Any]] = {}
            
            for res in keyword_results:
                docid = res["metadata"].get("docid", res.get("docid", str(uuid.uuid4())))
                keyword_dict[docid] = {
                    "content": res["content"],
                    "metadata": res["metadata"],
                    "keyword_score": res["keyword_score"],
                }

            all_doc_ids = set(list(semantic_dict.keys()) + list(keyword_dict.keys()))
            
            # Normalize
            max_sem = max((v.get("semantic_score", 0.0) for v in semantic_dict.values()), default=1.0)
            max_kw = max((v.get("keyword_score", 0.0) for v in keyword_dict.values()), default=1.0)

            hybrid_scores: Dict[str, Dict[str, Any]] = {}
            for docid in all_doc_ids:
                sem = semantic_dict.get(docid, {}).get("semantic_score", 0.0)
                kw = keyword_dict.get(docid, {}).get("keyword_score", 0.0)
                
                norm_sem = (sem / max_sem) if max_sem > 0 else 0.0
                norm_kw = (kw / max_kw) if max_kw > 0 else 0.0
                
                score = semantic_weight * norm_sem + keyword_weight * norm_kw
                
                hybrid_scores[docid] = {
                    "hybrid_score": score,
                    "semantic_score": sem,
                    "keyword_score": kw,
                }

            sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1]["hybrid_score"], reverse=True)
            
            formatted = []

            for docid, scores in sorted_results:
                if docid in semantic_dict:
                    doc = semantic_dict[docid]["doc"]
                    content = doc.page_content
                    metadata = dict(doc.metadata)
                    source = "semantic"
                elif docid in keyword_dict:
                    content = keyword_dict[docid]["content"]
                    metadata = dict(keyword_dict[docid]["metadata"])
                    source = "keyword"
                else:
                    continue
                
                formatted.append({
                    "content": content,
                    "metadata": metadata,
                    "hybrid_score": scores["hybrid_score"],
                    "semantic_score": scores["semantic_score"],
                    "keyword_score": scores["keyword_score"],
                    "source": metadata.get("source", "unknown"),
                    "search_type": "hybrid",
                    "primary_source": source
                })

            logger.info(f"Hybrid search completed: {len(formatted)} results from {collection_name}")
            
            return {
                "success": True,
                "results": formatted,
                "total": len(formatted),
                "query": query,
                "topk_used": k_actual,  
                "search_type": "hybrid",
                "keyword_weight": keyword_weight,
                "semantic_weight": semantic_weight,
                "total_chunks": total_chunks,
                "collection_name": collection_name,
            }
            
        except Exception as e:
            logger.error(f"Hybrid search error in {collection_name}: {e}")
            return {
                "success": False, 
                "error": str(e), 
                "results": [], 
                "total": 0, 
                "query": query,
                "collection_name": collection_name
            }

    def doc_exists(self, collection_name: str, filehash: str) -> bool:
        try:
            vs = self.get_collection(collection_name)
            data = vs.get(where={"filehash": filehash})
            exists = bool(data and data.get("ids"))
            logger.info(f"Document with hash {filehash[:8]}... exists in {collection_name}: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking doc existence in {collection_name}: {e}")
            return False

    def delete_by_hash(self, collection_name: str, filehash: str) -> bool:
        try:
            vs = self.get_collection(collection_name)
            results = vs.get(where={"filehash": filehash})
            ids = results.get("ids", []) if results else []
            
            if ids:
                vs.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} chunks for hash {filehash[:8]}... from {collection_name}")
                return True
            
            logger.warning(f"No documents found with hash {filehash[:8]}... in {collection_name}")
            return False
            
        except Exception as e:
            logger.error(f"Error deleting by hash from {collection_name}: {e}")
            return False

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        try:
            vs = self.get_collection(collection_name)
            data = vs.get()
            doc_count = len(data.get("ids", []))
            
            #user persist directory info
            user_persist_dir = self._get_user_persist_dir(collection_name)
            
            return {
                "name": collection_name,
                "doccount": doc_count,
                "exists": True,
                "persist_dir": str(user_persist_dir),
                "files_on_disk": list(user_persist_dir.iterdir()) if user_persist_dir.exists() else []
            }
        except Exception as e:
            return {
                "name": collection_name,
                "doccount": 0,
                "exists": False,
                "error": str(e)
            }

    def cleanup(self):
        self._collections.clear()
        logger.info("VectorStore cleaned up")

    def _tokenize(self, text: str):
        text = re.sub(r"[^a-z0-9\s]+", " ", text)
        return [t for t in text.split() if len(t) > 2]

    def _bm25_score(self, query_tokens, doc_text: str) -> float:
        doc_tokens = self._tokenize(doc_text)
        doc_len = len(doc_tokens)
        if doc_len == 0:
            return 0.0

        score = 0.0
        k1, b = 1.5, 0.75
        avgdl = 100.0

        for term in query_tokens:
            tf = doc_tokens.count(term)
            if tf == 0:
                continue
            idf = 1.0
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
            score += idf * (numerator / denominator)

        return score