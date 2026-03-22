"""
Temporal Parser Service for SocioMemory

Implements MemoTime-inspired temporal awareness for improved date/time handling:
1. Parse relative dates ("10 days ago", "last Friday") to absolute dates
2. Normalize temporal expressions for consistent storage and retrieval
3. Extract temporal constraints from queries for filtered search

Key insight from research: Temporal queries fail because:
- "10 days ago" in a query needs to match actual dates in memories
- Without date normalization, vector search can't handle temporal constraints
- Explicit temporal index enables date-range queries
"""
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TemporalType(Enum):
    """Type of temporal expression."""
    ABSOLUTE = "absolute"     # "January 15, 2024", "2024-01-15"
    RELATIVE = "relative"     # "yesterday", "last week", "3 days ago"
    ORDINAL = "ordinal"       # "first", "latest", "most recent"
    DURATION = "duration"     # "for 2 hours", "over 3 months"
    RANGE = "range"           # "between X and Y", "from X to Y"
    NONE = "none"


@dataclass
class TemporalInfo:
    """Parsed temporal information."""
    has_temporal: bool
    temporal_type: TemporalType
    reference_date: Optional[datetime] = None  # The parsed date if available
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    ordering_hint: Optional[str] = None  # "earliest", "latest", "specific"
    original_text: Optional[str] = None
    confidence: float = 1.0


class TemporalParser:
    """
    Parse and normalize temporal expressions in text.

    Inspired by MemoTime's temporal alignment approach:
    - Ground queries in temporal facts
    - Parse relative dates relative to "now" or a reference point
    - Enable temporal filtering in retrieval
    """

    # Day name to weekday number (Monday = 0)
    WEEKDAYS = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
        "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
    }

    # Month name to number
    MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
        "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    # Relative time patterns
    RELATIVE_PATTERNS = [
        # "X days/weeks/months/years ago"
        (r'(\d+)\s+(days?|weeks?|months?|years?)\s+ago', 'ago'),
        # "last week/month/year"
        (r'last\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)', 'last'),
        # "next week/month"
        (r'next\s+(week|month|year)', 'next'),
        # "yesterday", "today", "tomorrow"
        (r'\b(yesterday|today|tomorrow)\b', 'simple'),
        # "this week/month/year"
        (r'this\s+(week|month|year)', 'this'),
        # "in X days/weeks"
        (r'in\s+(\d+)\s+(days?|weeks?|months?)', 'future'),
    ]

    # Ordinal patterns (for "first", "latest", etc.)
    ORDINAL_PATTERNS = [
        (r'\b(first|earliest|initial|originally)\b', 'earliest'),
        (r'\b(last|latest|most\s+recent|recently|newest|final)\b', 'latest'),
        (r'\b(second|third|fourth|fifth)\b', 'specific'),
    ]

    # Absolute date patterns
    ABSOLUTE_PATTERNS = [
        # "January 15, 2024" or "January 15th, 2024"
        (r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})?', 'month_day'),
        # "15 January 2024" or "10 October, 2022" (with comma before year)
        # CRITICAL FIX: Added ,? to handle comma before year (benchmark format)
        (r'(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december),?\s*(\d{4})?', 'day_month'),
        # "2024-01-15" or "2024/01/15"
        (r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', 'iso'),
        # "01/15/2024" or "1/15/24"
        (r'(\d{1,2})/(\d{1,2})/(\d{2,4})', 'us_date'),
        # "on Monday", "on Friday"
        (r'on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', 'weekday'),
    ]

    def __init__(self, reference_time: Optional[datetime] = None):
        """
        Initialize the temporal parser.

        Args:
            reference_time: Reference time for relative date calculations.
                           Defaults to current time.
        """
        self.reference_time = reference_time or datetime.now(timezone.utc)

    def parse(self, text: str) -> TemporalInfo:
        """
        Parse temporal expressions from text.

        Args:
            text: Text that may contain temporal expressions

        Returns:
            TemporalInfo with parsed temporal data
        """
        text_lower = text.lower()

        # Check for ordinal patterns first (they affect how we interpret dates)
        ordering_hint = None
        for pattern, hint in self.ORDINAL_PATTERNS:
            if re.search(pattern, text_lower):
                ordering_hint = hint
                break

        # Try absolute date patterns
        for pattern, pattern_type in self.ABSOLUTE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                parsed_date = self._parse_absolute_match(match, pattern_type)
                if parsed_date:
                    return TemporalInfo(
                        has_temporal=True,
                        temporal_type=TemporalType.ABSOLUTE,
                        reference_date=parsed_date,
                        ordering_hint=ordering_hint or "specific",
                        original_text=match.group(0),
                        confidence=0.9,
                    )

        # Try relative date patterns
        for pattern, pattern_type in self.RELATIVE_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                parsed_date = self._parse_relative_match(match, pattern_type)
                if parsed_date:
                    return TemporalInfo(
                        has_temporal=True,
                        temporal_type=TemporalType.RELATIVE,
                        reference_date=parsed_date,
                        ordering_hint=ordering_hint or "specific",
                        original_text=match.group(0),
                        confidence=0.8,
                    )

        # If only ordinal pattern found
        if ordering_hint:
            return TemporalInfo(
                has_temporal=True,
                temporal_type=TemporalType.ORDINAL,
                ordering_hint=ordering_hint,
                confidence=0.7,
            )

        # No temporal expression found
        return TemporalInfo(
            has_temporal=False,
            temporal_type=TemporalType.NONE,
        )

    def _parse_absolute_match(self, match: re.Match, pattern_type: str) -> Optional[datetime]:
        """Parse an absolute date match."""
        try:
            if pattern_type == 'month_day':
                month_str, day, year = match.groups()
                month = self.MONTHS.get(month_str.lower())
                day = int(day)
                year = int(year) if year else self.reference_time.year
                return datetime(year, month, day, tzinfo=timezone.utc)

            elif pattern_type == 'day_month':
                day, month_str, year = match.groups()
                month = self.MONTHS.get(month_str.lower())
                day = int(day)
                year = int(year) if year else self.reference_time.year
                return datetime(year, month, day, tzinfo=timezone.utc)

            elif pattern_type == 'iso':
                year, month, day = match.groups()
                return datetime(int(year), int(month), int(day), tzinfo=timezone.utc)

            elif pattern_type == 'us_date':
                month, day, year = match.groups()
                year = int(year)
                if year < 100:
                    year += 2000 if year < 50 else 1900
                return datetime(year, int(month), int(day), tzinfo=timezone.utc)

            elif pattern_type == 'weekday':
                weekday_str = match.group(1).lower()
                target_weekday = self.WEEKDAYS.get(weekday_str)
                if target_weekday is not None:
                    return self._get_weekday_date(target_weekday, past=True)

        except (ValueError, TypeError):
            pass

        return None

    def _parse_relative_match(self, match: re.Match, pattern_type: str) -> Optional[datetime]:
        """Parse a relative date match."""
        try:
            if pattern_type == 'ago':
                amount = int(match.group(1))
                unit = match.group(2).rstrip('s')  # Remove plural 's'
                return self._subtract_time(amount, unit)

            elif pattern_type == 'last':
                unit = match.group(1).lower()
                if unit in self.WEEKDAYS:
                    return self._get_weekday_date(self.WEEKDAYS[unit], past=True)
                elif unit == 'week':
                    return self._subtract_time(1, 'week')
                elif unit == 'month':
                    return self._subtract_time(1, 'month')
                elif unit == 'year':
                    return self._subtract_time(1, 'year')

            elif pattern_type == 'next':
                unit = match.group(1).lower()
                if unit == 'week':
                    return self._add_time(1, 'week')
                elif unit == 'month':
                    return self._add_time(1, 'month')
                elif unit == 'year':
                    return self._add_time(1, 'year')

            elif pattern_type == 'simple':
                word = match.group(1).lower()
                if word == 'yesterday':
                    return self._subtract_time(1, 'day')
                elif word == 'today':
                    return self.reference_time.replace(hour=0, minute=0, second=0, microsecond=0)
                elif word == 'tomorrow':
                    return self._add_time(1, 'day')

            elif pattern_type == 'this':
                # "this week/month" - return start of current period
                unit = match.group(1).lower()
                now = self.reference_time
                if unit == 'week':
                    # Start of current week (Monday)
                    days_since_monday = now.weekday()
                    return now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
                elif unit == 'month':
                    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                elif unit == 'year':
                    return now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

            elif pattern_type == 'future':
                amount = int(match.group(1))
                unit = match.group(2).rstrip('s')
                return self._add_time(amount, unit)

        except (ValueError, TypeError):
            pass

        return None

    def _subtract_time(self, amount: int, unit: str) -> datetime:
        """Subtract time from reference date."""
        if unit == 'day':
            return self.reference_time - timedelta(days=amount)
        elif unit == 'week':
            return self.reference_time - timedelta(weeks=amount)
        elif unit == 'month':
            # Approximate month subtraction
            new_month = self.reference_time.month - amount
            new_year = self.reference_time.year
            while new_month <= 0:
                new_month += 12
                new_year -= 1
            return self.reference_time.replace(year=new_year, month=new_month)
        elif unit == 'year':
            return self.reference_time.replace(year=self.reference_time.year - amount)
        return self.reference_time

    def _add_time(self, amount: int, unit: str) -> datetime:
        """Add time to reference date."""
        if unit == 'day':
            return self.reference_time + timedelta(days=amount)
        elif unit == 'week':
            return self.reference_time + timedelta(weeks=amount)
        elif unit == 'month':
            new_month = self.reference_time.month + amount
            new_year = self.reference_time.year
            while new_month > 12:
                new_month -= 12
                new_year += 1
            return self.reference_time.replace(year=new_year, month=new_month)
        elif unit == 'year':
            return self.reference_time.replace(year=self.reference_time.year + amount)
        return self.reference_time

    def _get_weekday_date(self, target_weekday: int, past: bool = True) -> datetime:
        """Get the date of a specific weekday (past or future)."""
        current_weekday = self.reference_time.weekday()

        if past:
            # Find the most recent occurrence of this weekday
            days_back = (current_weekday - target_weekday) % 7
            if days_back == 0:
                days_back = 7  # If today, go back a week
            return self.reference_time - timedelta(days=days_back)
        else:
            # Find the next occurrence of this weekday
            days_forward = (target_weekday - current_weekday) % 7
            if days_forward == 0:
                days_forward = 7  # If today, go forward a week
            return self.reference_time + timedelta(days=days_forward)

    def normalize_date_in_text(self, text: str) -> Tuple[str, Optional[datetime]]:
        """
        Normalize relative dates in text to absolute dates.

        Useful for memory ingestion - converts "yesterday" to "December 25, 2025"
        so that future queries can match.

        Args:
            text: Text containing potential relative dates

        Returns:
            Tuple of (text with normalized date, parsed datetime)
        """
        info = self.parse(text)

        if not info.has_temporal or not info.reference_date:
            return text, None

        # Format the normalized date
        normalized = info.reference_date.strftime("%B %d, %Y")

        # Replace the original text with normalized date
        if info.original_text:
            # Add both the original and normalized for better matching
            new_text = text.replace(
                info.original_text,
                f"{info.original_text} ({normalized})"
            )
            return new_text, info.reference_date

        return text, info.reference_date

    def get_date_range_for_query(
        self,
        query: str,
        buffer_days: int = 1
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get a date range for filtering memories based on a temporal query.

        Args:
            query: Query text with temporal expression
            buffer_days: Days to add before/after the parsed date for fuzzy matching

        Returns:
            Tuple of (start_date, end_date) for filtering, or (None, None)
        """
        info = self.parse(query)

        if not info.has_temporal:
            return None, None

        # For ordinal queries ("first", "latest"), don't filter by date
        # Instead, let sorting handle it
        if info.temporal_type == TemporalType.ORDINAL:
            return None, None

        if info.reference_date:
            buffer = timedelta(days=buffer_days)
            start = info.reference_date - buffer
            end = info.reference_date + buffer
            return start, end

        return None, None


# Singleton instance
_parser: Optional[TemporalParser] = None


def get_temporal_parser(reference_time: Optional[datetime] = None) -> TemporalParser:
    """Get or create singleton temporal parser."""
    global _parser
    if _parser is None or reference_time is not None:
        _parser = TemporalParser(reference_time)
    return _parser
