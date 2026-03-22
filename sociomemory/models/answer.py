"""
Pydantic models for Answer endpoint
"""
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class QuestionTypeInfo(BaseModel):
    """Detected question type flags"""
    is_aggregation: bool = False
    is_temporal: bool = False
    is_knowledge_update: bool = False
    is_preference: bool = False
    is_complex: bool = False
    asks_old_value: bool = False


class AnswerRequest(BaseModel):
    """Request model for generating an answer from memories"""
    user_id: UUID
    question: str = Field(..., min_length=1, max_length=2000)
    question_date: Optional[str] = Field(None, description="Today's date for temporal reasoning (YYYY-MM-DD)")
    search_limit: int = Field(default=10, ge=1, le=100)
    search_mode: str = Field(default="hyper", description="Search mode: hyper, standard, etc.")
    enable_voting: bool = Field(default=True, description="Enable self-consistency voting for complex questions")
    search_results: Optional[list[dict[str, Any]]] = Field(
        None, description="Pre-computed search results. If provided, skips search and uses these directly."
    )


class AnswerResponse(BaseModel):
    """Response model for generated answers"""
    answer: str
    question_type: QuestionTypeInfo
    search_results_count: int
    voting_used: bool
    duration_ms: int
    crag_queries: Optional[list[str]] = Field(None, description="CRAG queries generated when answer was insufficient")
    crag_iteration: int = Field(0, description="0 = no CRAG, 1 = used CRAG re-search")
