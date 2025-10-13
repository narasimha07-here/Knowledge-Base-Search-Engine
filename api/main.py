import os
import time
import asyncio
import aiofiles
import logging
from pathlib import Path
from typing import List
from datetime import datetime
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import FastAPI,HTTPException,UploadFile,File,Form,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

from pydantic_models import (
    UserCreate,
    UserResponse,
    GenericResponse,
    FileUploadResponse,
    DocumentResponse,
    SearchQuery,
    HybridQuery,
    SearchHistory,
    UserStats,
    HybridResult,
)
from db_utils import DbManager
from chroma_utils import VectorStore
from langchain_utils import LLMManager
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="RAG Search API",
    description="Document search",
    version="1.0.1",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.getenv("DBPATH")
UPLOAD_DIR = Path(os.getenv("UPLOADDIR"))
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(1000 * 1024 * 1024)))  

db = DbManager(DB_PATH)
vs = VectorStore(persist_dir=os.getenv("VECTORDIR"), model=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
llm = LLMManager()


def user_upload_dir(user_id: int) -> Path:
    p = UPLOAD_DIR / f"user_{user_id}"
    p.mkdir(exist_ok=True, parents=True)
    return p

def user_collection_name(user_id: int) -> str:
    return vs.user_collection_name(user_id)

async def run_blocking(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def process_doc_bg(filepath: str, user_id: int, filename: str, filehash: str, doc_id: int):
    try:

        docs = await llm.load_file(filepath)
        collection = user_collection_name(user_id)
        exists = await run_blocking(vs.doc_exists, collection, filehash)
        if not exists:
            res = await run_blocking(vs.add_docs, collection, docs, filehash)
            logger.info(f"Doc processing done for {filename}: {res}")
        else:
            logger.info(f"Doc {filename} already exists in vector store, skipping")
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")


@app.get("/", response_model=GenericResponse)
async def root():
    return GenericResponse(message="RAG Search API is running!")

ph = PasswordHasher()

@app.post("/register", response_model=UserResponse)
async def register(userdata: UserCreate):
    try:
        # Hash the raw password directly with Argon2
        hashedpwd = ph.hash(userdata.password)
        
        # Create user in the database with the new hash
        user_id = db.create_user(
            username=userdata.username, 
            email=userdata.email, 
            hashedpwd=hashedpwd
        )
        
        user = db.get_user_by_id(user_id)
        return UserResponse(
            id=user["id"],
            username=user["username"],
            email=user["email"],
            created_at=datetime.fromisoformat(user["created_at"]),
            is_active=bool(user["is_active"]),
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # logger.exception("Register failed")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/login", response_model=UserResponse)
async def login(username: str = Form(...), password: str = Form(...)):
    try:
        userdata = db.get_user_by_name(username)
        if not userdata:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Verify the raw password against the stored Argon2 hash
        try:
            ph.verify(userdata["hashed_password"], password)
        except VerifyMismatchError:
            # This error means the password does not match
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Optional: Check if the hash parameters need to be updated
        if ph.check_needs_rehash(userdata["hashed_password"]):
            new_hash = ph.hash(password)
            db.update_user_password(userdata["id"], new_hash)

        db.update_user_activity(userdata["id"])
        return UserResponse(
            id=userdata["id"],
            username=userdata["username"],
            email=userdata["email"],
            created_at=datetime.fromisoformat(userdata["created_at"]),
            is_active=bool(userdata["is_active"]),
        )
        
    except HTTPException:
        raise
    except Exception:
        # logger.exception("Login failed")
        raise HTTPException(status_code=500, detail="Login failed")


@app.post("/upload", response_model=FileUploadResponse)
async def upload_files(
    bg_tasks: BackgroundTasks,
    userid: int = Form(...),
    files: List[UploadFile] = File(...),
):
    uploaded = []
    failed = []
    successcount = 0

    for file in files:
        try:
            if not llm.is_supported_file(file.filename):
                failed.append({"filename": file.filename, "error": f"Unsupported type. Supported: {', '.join(llm.supported_types)}"})
                continue

            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                failed.append({"filename": file.filename, "error": f"File too large. Max {MAX_FILE_SIZE // (1024 * 1024)}MB"})
                continue

            filehash = db.calc_hash(content)
            existing = db.get_doc_by_hash(userid, filehash)
            if existing:
                uploaded.append({
                    "filename": file.filename,
                    "document_id": existing["id"],
                    "filehash": filehash,
                    "status": "exists",
                    "message": "File already exists for this user"
                })
                successcount += 1
                continue

            udir = user_upload_dir(userid)
            safe_name = f"{filehash}_{Path(file.filename).name}"
            filepath = udir / safe_name

            async with aiofiles.open(filepath, "wb") as f:
                await f.write(content)



            doc_id = db.create_doc(
                userid=userid,
                filename=file.filename,
                filepath=str(filepath),
                filehash=filehash,
                filesize=len(content),
                filetype=Path(file.filename).suffix.lower(),
                vector_id=None,
            )

            bg_tasks.add_task(process_doc_bg, str(filepath), userid, file.filename, filehash, doc_id)

            uploaded.append({
                "filename": file.filename,
                "document_id": doc_id,
                "filehash": filehash,
                "status": "uploaded",
                "message": "File uploaded, processing..."
            })
            successcount += 1
        except Exception as e:
            logger.exception(f"Upload error for {file.filename}")
            failed.append({"filename": file.filename, "error": f"Upload error: {str(e)}"})

    return FileUploadResponse(
        message=f"Processed {len(files)} files: {successcount} ok, {len(failed)} failed.",
        uploaded_files=uploaded,
        total_processed=len(files),
        successful_uploads=successcount,
        failed_uploads=failed,
        success=len(failed) == 0
    )

@app.get("/documents", response_model=List[DocumentResponse])
async def list_docs(userid: int, limit: int = 50, offset: int = 0):
    try:
        docs = db.get_user_docs(userid=userid, limit=limit, offset=offset)
        return [
            DocumentResponse(
                id=doc["id"],
                userid=doc["userid"],
                filename=doc["filename"],
                filehash=doc["filehash"],
                filesize=doc["filesize"],
                filetype=doc["filetype"],
                uploaddate=datetime.fromisoformat(doc["upload_date"]),
                vectorstoreid=doc["vectorstore_id"],
            )
            for doc in docs
        ]
    except Exception as e:
        logger.exception("Error listing docs")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.delete("/documents/{docid}", response_model=GenericResponse)
async def delete_doc(docid: int, userid: int):
    try:
        user_docs = db.get_user_docs(userid=userid, limit=1000, offset=0)
        doc = None
        for d in user_docs:
            if d["id"] == docid:
                doc = d
                break
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        collection = user_collection_name(userid)
        _ = await run_blocking(vs.delete_by_hash, collection, doc["filehash"])

        success = db.delete_doc(docid, userid)
        if success:
            try:
                Path(doc["filepath"]).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Couldn't delete file: {e}")
            return GenericResponse(message="Document deleted")
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Delete error")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/search/semantic")
async def semantic_search(query: SearchQuery):
    start_time = time.time()
    try:
        collection = user_collection_name(query.userid)
        vectorstore = await run_blocking(vs.get_collection, collection)
        retriever = vectorstore.as_retriever(search_kwargs={"k": query.topk})
        result = await llm.generate_response(retriever=retriever, question=query.query)
        search_time = time.time() - start_time
        db.save_search(userid=query.userid, query=query.query, results_count=result["retrieved_chunks"], search_time=search_time)
        return {
            "query": query.query,
            "response": result["response"],
            "sources": result["sources"],
            "search_time": search_time,
            "retrieved_chunks": result["retrieved_chunks"],
            "success": result["success"]
        }
    except Exception as e:
        logger.exception("Semantic search error")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/search/hybrid", response_model=HybridResult)
async def hybrid_search(query: HybridQuery):
    start_time = time.time()
    try:
        collection = user_collection_name(query.userid)

        hybrid_result = await run_blocking(
            vs.hybrid_search,
            collection,
            query.query,
            None if query.use_all_chunks else (query.topk or 10),
            query.keyword_weight,
            query.semantic_weight,
        )
        if not hybrid_result["success"]:
            raise HTTPException(status_code=500, detail=hybrid_result.get("error", "Hybrid search failed"))

        retrieved_docs: List[Document] = []
        for it in hybrid_result["results"]:
            doc = Document(page_content=it["content"], metadata=it["metadata"])
            doc.metadata.update({
                "search_type": it.get("search_type", "hybrid"),
                "hybrid_score": it.get("hybrid_score", 0.0),
                "semantic_score": it.get("semantic_score", 0.0),
                "keyword_score": it.get("keyword_score", 0.0),
            })
            retrieved_docs.append(doc)

        vectorstore = await run_blocking(vs.get_collection, collection)
        rag_result = await llm.generate_hybrid_response(
            retriever=vectorstore.as_retriever(),
            question=query.query,
            hybrid_docs_func=lambda q: retrieved_docs
        )

        search_time = time.time() - start_time
        db.save_search(userid=query.userid, query=f"HYBRID: {query.query}", results_count=rag_result["retrieved_chunks"], search_time=search_time)

        return HybridResult(
            query=query.query,
            response=rag_result["response"],
            sources=rag_result["sources"],
            search_time=search_time,
            retrieved_chunks=rag_result["retrieved_chunks"],
            search_type="hybrid",
            keyword_weight=query.keyword_weight,
            semantic_weight=query.semantic_weight,
            hybrid_results=hybrid_result["results"],
            topk_used=hybrid_result.get("topk_used", len(hybrid_result["results"])),
            success=rag_result["success"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Hybrid search error")
        raise HTTPException(status_code=500, detail=f"Hybrid search error: {str(e)}")

@app.get("/search/history", response_model=List[SearchHistory])
async def get_history(userid: int, limit: int = 50, offset: int = 0):
    try:
        history = db.get_search_history(userid=userid, limit=limit, offset=offset)
        return [
            SearchHistory(
                id=rec["id"],
                userid=rec["userid"],
                query=rec["query"],
                results_count=rec["results_count"],
                search_time=rec["search_time"],
                created_at=datetime.fromisoformat(rec["created_at"]),
            )
            for rec in history
        ]
    except Exception as e:
        logger.exception("History error")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/stats", response_model=UserStats)
async def get_stats(userid: int):
    try:
        stats = db.get_stats(userid)
        return UserStats(
            userid=stats["userid"],
            total_documents=stats["total_documents"],
            total_searches=stats["total_searches"],
            storage_used_mb=stats["storage_used_mb"],
            last_activity=datetime.fromisoformat(stats["last_activity"]) if stats["last_activity"] else None
        )
    except Exception as e:
        logger.exception("Stats error")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.1", "features": ["semantic_search", "keyword_search", "hybrid_search"]}
