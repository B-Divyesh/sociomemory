"""
FSRS (Free Spaced Repetition Scheduler) integration for memory retrieval optimization.

This module adapts FSRS concepts for memory prioritization:
- Stability: How well-established a memory is
- Difficulty: How hard the memory is to retrieve
- Retrievability: Current probability of successful recall
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
import math

from fsrs import Card, Scheduler, Rating, State

from sociomemory.config import get_settings


@dataclass
class MemoryFSRSState:
    """FSRS state for a memory."""
    stability: float  # Days until 90% retention probability
    difficulty: float  # 0-10, higher = harder
    retrievability: float  # Current recall probability (0-1)
    last_accessed: Optional[datetime] = None
    access_count: int = 0


class FSRSScheduler:
    """
    Adapts FSRS for memory retrieval prioritization.

    Instead of scheduling reviews, we use FSRS to:
    1. Track memory strength (stability)
    2. Estimate recall probability (retrievability)
    3. Prioritize memories for retrieval
    """

    def __init__(self, desired_retention: float = 0.9, max_interval: int = 365):
        """
        Initialize FSRS scheduler.

        Args:
            desired_retention: Target retention rate (default 90%)
            max_interval: Maximum interval in days
        """
        self.scheduler = Scheduler(
            desired_retention=desired_retention,
            maximum_interval=max_interval,
            enable_fuzzing=False,  # Disable fuzzing for deterministic results
        )
        self.desired_retention = desired_retention

    def create_initial_state(self) -> MemoryFSRSState:
        """Create initial FSRS state for a new memory."""
        card = Card()
        now = datetime.now(timezone.utc)

        return MemoryFSRSState(
            stability=card.stability if card.stability else 4.0,  # Default 4.0 days stability
            difficulty=card.difficulty if card.difficulty else 0.3,  # Default 0.3 difficulty (matches DB schema)
            retrievability=1.0,  # New memory has perfect retrievability
            last_accessed=now,
            access_count=0,
        )

    def update_on_access(
        self,
        current_state: MemoryFSRSState,
        was_useful: bool,
        access_time: Optional[datetime] = None
    ) -> MemoryFSRSState:
        """
        Update FSRS state when a memory is accessed.

        Args:
            current_state: Current FSRS state
            was_useful: Whether the memory was useful in the response
            access_time: Time of access (defaults to now)

        Returns:
            Updated FSRS state
        """
        access_time = access_time or datetime.now(timezone.utc)
        if access_time.tzinfo is None:
            access_time = access_time.replace(tzinfo=timezone.utc)

        # Create a card from current state
        card = Card()
        card.stability = current_state.stability
        card.difficulty = current_state.difficulty

        # Set last_review if we have it
        if current_state.last_accessed:
            last_accessed = current_state.last_accessed
            if last_accessed.tzinfo is None:
                last_accessed = last_accessed.replace(tzinfo=timezone.utc)
            card.last_review = last_accessed

        # Map usefulness to rating
        # Good = memory was useful, Again = memory wasn't useful
        rating = Rating.Good if was_useful else Rating.Again

        # Review the card
        updated_card, _ = self.scheduler.review_card(
            card=card,
            rating=rating,
            review_datetime=access_time,
        )

        # Calculate current retrievability
        retrievability = self.scheduler.get_card_retrievability(
            updated_card,
            current_datetime=access_time,
        )

        return MemoryFSRSState(
            stability=updated_card.stability,
            difficulty=updated_card.difficulty,
            retrievability=retrievability,
            last_accessed=access_time,
            access_count=current_state.access_count + 1,
        )

    def calculate_retrievability(
        self,
        state: MemoryFSRSState,
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Calculate current retrievability for a memory.

        Uses the FSRS formula: R = e^(-t/S)
        where t is time since last access and S is stability.

        Args:
            state: Current FSRS state
            current_time: Time to calculate for (defaults to now)

        Returns:
            Retrievability (0-1)
        """
        current_time = current_time or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        if not state.last_accessed:
            return 1.0  # Never accessed = full retrievability

        last_accessed = state.last_accessed
        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)

        # Calculate days since last access
        elapsed = (current_time - last_accessed).total_seconds() / 86400.0

        if elapsed <= 0:
            return 1.0

        # FSRS retrievability formula: R = e^(-t/S)
        # where t is time in days and S is stability
        stability = max(state.stability, 0.1)  # Prevent division issues
        retrievability = math.exp(-elapsed / stability)

        return max(0.0, min(1.0, retrievability))

    def get_priority_score(
        self,
        state: MemoryFSRSState,
        similarity: float,
        confidence: float,
        current_time: Optional[datetime] = None
    ) -> float:
        """
        Calculate priority score for memory retrieval.

        Combines:
        - Similarity (50%): Vector similarity to query
        - Retrievability (30%): FSRS-based memory strength
        - Confidence (10%): Memory reliability score
        - Recency (10%): How recently accessed

        Args:
            state: FSRS state
            similarity: Vector similarity (0-1)
            confidence: Memory confidence (0-1)
            current_time: Current time

        Returns:
            Priority score (0-1)
        """
        current_time = current_time or datetime.now(timezone.utc)

        # Get current retrievability
        retrievability = self.calculate_retrievability(state, current_time)

        # Calculate recency score (1 for recent, decays over time)
        recency = 1.0
        if state.last_accessed:
            last_accessed = state.last_accessed
            if last_accessed.tzinfo is None:
                last_accessed = last_accessed.replace(tzinfo=timezone.utc)
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)

            days_ago = (current_time - last_accessed).total_seconds() / 86400.0
            recency = math.exp(-days_ago / 30)  # Decay over 30 days

        # Weighted combination
        priority = (
            similarity * 0.50 +
            retrievability * 0.30 +
            confidence * 0.10 +
            recency * 0.10
        )

        return max(0.0, min(1.0, priority))


# Singleton instance
_scheduler: Optional[FSRSScheduler] = None


def get_fsrs_scheduler() -> FSRSScheduler:
    """Get or create the FSRS scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        settings = get_settings()
        _scheduler = FSRSScheduler(
            desired_retention=settings.fsrs_desired_retention,
            max_interval=settings.fsrs_max_interval,
        )
    return _scheduler
