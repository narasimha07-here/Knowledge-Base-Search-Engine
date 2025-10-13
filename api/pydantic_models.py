from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict
from datetime import datetime

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    is_active: bool

class DocumentBase(BaseModel):
    filename: str
    filesize: int
    filetype: str

class DocumentCreate(DocumentBase):
    userid: int
    filehash: str
    filepath: str

class DocumentResponse(DocumentBase):
    id: int
    userid: int
    filehash: str
    uploaddate: datetime
    vectorstoreid: Optional[str] = None

    class Config:
        from_attributes = True

class FileUploadResponse(BaseModel):
    message: str
    uploaded_files: List[Dict]
    total_processed: int
    successful_uploads: int
    failed_uploads: List[Dict]
    success: bool = True

class SearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    userid: int
    topk: int = Field(default=10, ge=1, le=50)

class HybridQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    userid: int
    topk: Optional[int] = Field(default=None, ge=1)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    use_all_chunks: bool = Field(default=True)

class HybridResult(BaseModel):
    query: str
    response: str
    sources: List[Dict]
    search_time: float
    retrieved_chunks: int
    search_type: str
    keyword_weight: float
    semantic_weight: float
    hybrid_results: List[Dict]
    topk_used: int
    success: bool

class SearchHistory(BaseModel):
    id: int
    userid: int
    query: str
    results_count: int
    search_time: float
    created_at: datetime

    class Config:
        from_attributes = True

class GenericResponse(BaseModel):
    message: str
    success: bool = True

class UserStats(BaseModel):
    userid: int
    total_documents: int
    total_searches: int
    storage_used_mb: float
    last_activity: Optional[datetime]
