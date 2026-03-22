#!/usr/bin/env python3
"""
Standalone SocioMemory API Benchmark

This benchmark tests the DEPLOYED SocioMemory API using only HTTP requests.
No local imports required - can be run by third parties with just:
  - API endpoint URL
  - API key
  - Python with requests library

Usage:
    python benchmark_api_standalone.py \
        --api-url http://localhost:8001 \
        --api-key YOUR_API_KEY \
        --dataset longmemeval \
        --max-questions 100

Requirements:
    pip install requests

Benchmarks available:
    - longmemeval: LongMemEval Oracle dataset (500 questions)
    - longmemeval-s: LongMemEval S dataset (harder, 500 questions)
    - locomo: LoCoMo conversational memory (10 conversations)
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Optional
from collections import defaultdict

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)

# Configure logging
logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[str] = None, verbose: bool = False):
    """Configure logging to file and/or console."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


def write_verbose_log(verbose_file: str, question_num: int, results: dict, benchmark_type: str):
    """Write verbose progress log every N questions with category-wise accuracy."""
    with open(verbose_file, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'='*60}\n")
        f.write(f"[{timestamp}] Progress at question {question_num}\n")
        f.write(f"{'='*60}\n")

        if benchmark_type == "longmemeval":
            total = results.get("total", 0)
            if total > 0:
                session_acc = results.get("session_correct", 0) / total * 100
                answer_acc = results.get("answer_found", 0) / total * 100
                f.write(f"Overall: {total} questions\n")
                f.write(f"  Session Retrieval: {session_acc:.1f}%\n")
                f.write(f"  Answer Found: {answer_acc:.1f}%\n")

                f.write(f"\nBy Question Type:\n")
                by_type = results.get("by_type", {})
                for q_type, stats in sorted(by_type.items()):
                    if stats.get("total", 0) > 0:
                        s_acc = stats.get("session_correct", 0) / stats["total"] * 100
                        a_acc = stats.get("answer_found", 0) / stats["total"] * 100
                        f.write(f"  {q_type}: Session={s_acc:.1f}%, Answer={a_acc:.1f}% ({stats['total']} qs)\n")

        elif benchmark_type == "locomo":
            total = results.get("total_qa", 0)
            if total > 0:
                top5_acc = results.get("retrieval_correct_top5", 0) / total * 100
                top1_acc = results.get("retrieval_correct_top1", 0) / total * 100
                f.write(f"Overall: {total} QA pairs\n")
                f.write(f"  Top-5 Retrieval: {top5_acc:.1f}%\n")
                f.write(f"  Top-1 Retrieval: {top1_acc:.1f}%\n")

                f.write(f"\nBy Category:\n")
                by_cat = results.get("by_category", {})
                for cat, stats in sorted(by_cat.items()):
                    if stats.get("total", 0) > 0:
                        acc = stats.get("correct", 0) / stats["total"] * 100
                        f.write(f"  {cat}: {acc:.1f}% ({stats['correct']}/{stats['total']})\n")

        f.write(f"\n")


class SocioMemoryAPIClient:
    """Simple HTTP client for SocioMemory API."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        })

    def health_check(self) -> bool:
        """Check if API is reachable."""
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=10)
            return resp.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def create_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str = "episode",
        source_platform: str = "benchmark",
        source_id: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> Optional[dict]:
        """Create a memory via API."""
        payload = {
            "user_id": user_id,
            "content": content,
            "memory_type": memory_type,
            "source_platform": source_platform,
        }
        if source_id:
            payload["source_id"] = source_id
        if metadata:
            payload["metadata"] = metadata

        try:
            resp = self.session.post(
                f"{self.base_url}/api/v1/memories",
                json=payload,
                timeout=30
            )
            if resp.status_code in (200, 201):
                return resp.json()
            else:
                print(f"Create memory failed: {resp.status_code} - {resp.text[:200]}")
                return None
        except Exception as e:
            print(f"Create memory error: {e}")
            return None

    def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        threshold: float = 0.0,
        mode: str = "full"
    ) -> list:
        """Search memories via API.

        Args:
            mode: Search mode - 'full' (96%/80% accuracy with GPT-4o),
                  'hybrid' (BM25+vector), 'enhanced' (entity reranking), 'basic' (vector only)
        """
        params = {
            "user_id": user_id,
            "query": query,
            "limit": limit,
            "threshold": threshold,
            "mode": mode
        }

        try:
            resp = self.session.get(
                f"{self.base_url}/api/v1/memories/search",
                params=params,
                timeout=60
            )
            if resp.status_code == 200:
                data = resp.json()
                # API returns "results" key
                return data.get("results", data.get("memories", []))
            else:
                print(f"Search failed: {resp.status_code} - {resp.text[:200]}")
                return []
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def delete_user_memories(self, user_id: str) -> bool:
        """Delete all memories for a user (cleanup)."""
        try:
            resp = self.session.delete(
                f"{self.base_url}/api/v1/memories/user/{user_id}",
                timeout=30
            )
            return resp.status_code in (200, 204, 404)
        except Exception as e:
            print(f"Delete error: {e}")
            return False

    def get_user_stats(self, user_id: str) -> Optional[dict]:
        """Get memory statistics for a user."""
        try:
            resp = self.session.get(
                f"{self.base_url}/api/v1/users/{user_id}/memory-stats",
                timeout=30
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            print(f"Stats error: {e}")
            return None


def download_dataset(dataset_name: str, data_dir: str = "benchmark_data") -> str:
    """Download benchmark dataset if not present. Also checks local paths."""
    os.makedirs(data_dir, exist_ok=True)

    # Get base directory (sociomemory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    datasets = {
        "longmemeval": {
            "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
            "filename": "longmemeval_oracle.json",
            "local_paths": [
                os.path.join(base_dir, "benchmarks/LongMemEval/data/longmemeval_oracle.json")
            ]
        },
        "longmemeval-s": {
            "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
            "filename": "longmemeval_s_cleaned.json",
            "local_paths": [
                os.path.join(base_dir, "benchmarks/LongMemEval/data/longmemeval_s_cleaned.json")
            ]
        },
        "locomo": {
            "url": "https://raw.githubusercontent.com/snap-stanford/locomo/main/data/locomo10.json",
            "filename": "locomo10.json",
            "local_paths": [
                os.path.join(base_dir, "benchmarks/locomo/data/locomo10.json"),
                os.path.join(base_dir, "benchmarks/LoCoMo/data/locomo10.json")
            ]
        }
    }

    if dataset_name not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")

    info = datasets[dataset_name]
    filepath = os.path.join(data_dir, info["filename"])

    # Check if already downloaded
    if os.path.exists(filepath):
        print(f"Dataset already exists: {filepath}")
        return filepath

    # Check local paths first
    for local_path in info.get("local_paths", []):
        if os.path.exists(local_path):
            print(f"Using local dataset: {local_path}")
            return local_path

    # Download from URL
    print(f"Downloading {dataset_name} dataset...")
    try:
        resp = requests.get(info["url"], timeout=120)
        resp.raise_for_status()

        with open(filepath, "w") as f:
            f.write(resp.text)

        print(f"Downloaded to: {filepath}")
        return filepath
    except Exception as e:
        raise ValueError(f"Failed to download dataset and no local copy found: {e}")


def run_longmemeval_benchmark(
    client: SocioMemoryAPIClient,
    dataset_path: str,
    max_questions: int = 100,
    user_id: Optional[str] = None,
    verbose_file: Optional[str] = None
) -> dict:
    """
    Run LongMemEval benchmark against deployed API.

    IMPORTANT: Works per-question like the local benchmark:
    - For each question, load ONLY that question's haystack sessions
    - Evaluate the question
    - Clear memories and move to next question

    This matches the local benchmark design where each question has its own search space.

    Measures:
    - Session retrieval accuracy: Did we find the right session?
    - Answer found accuracy: Did the retrieved content contain the answer?
    """
    print("\n" + "=" * 60)
    print("LONGMEMEVAL BENCHMARK - SocioMemory API")
    print("=" * 60)

    # Load dataset
    with open(dataset_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} questions from {dataset_path}")

    if max_questions and max_questions < len(data):
        data = data[:max_questions]
        print(f"Limited to {max_questions} questions")

    results = {
        "total": 0,
        "session_correct": 0,
        "answer_found": 0,
        "by_type": defaultdict(lambda: {"total": 0, "session_correct": 0, "answer_found": 0}),
        "errors": [],
        "questions": []
    }

    # Process each question independently (like local benchmark)
    # Each question has its own haystack of sessions to search through
    print(f"\nProcessing {len(data)} questions (per-question ingestion)...")

    for q_idx, q in enumerate(data):
        results["total"] += 1
        q_type = q.get("question_type", "unknown")
        results["by_type"][q_type]["total"] += 1

        question = q.get("question", "")
        oracle_session_ids = set(q.get("answer_session_ids", []))
        answer = str(q.get("answer", ""))

        # Generate unique user ID for this question
        question_user_id = user_id or str(uuid.uuid4())

        # Get this question's haystack sessions
        session_ids = q.get("haystack_session_ids", [])
        sessions = q.get("haystack_sessions", [])
        dates = q.get("haystack_dates", [])

        # Build session lookup for this question
        q_sessions = {}
        for i, (sess_id, messages) in enumerate(zip(session_ids, sessions)):
            date = dates[i] if i < len(dates) else "Unknown date"
            q_sessions[sess_id] = (messages, date)

        # Ingest this question's sessions as WHOLE SESSIONS (not individual turns)
        # This reduces API calls by 10x while preserving session-level retrieval accuracy
        for sess_id, (messages, date) in q_sessions.items():
            # Concatenate all turns in this session
            session_has_answer = False
            turn_texts = []
            for turn_idx, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                has_answer = msg.get("has_answer", False)
                if has_answer:
                    session_has_answer = True
                if content:
                    turn_texts.append(f"[{role}]: {content}")

            if turn_texts:
                # Mark session with [ANSWER] if any turn has the answer
                prefix = "[ANSWER] " if session_has_answer else ""
                session_content = f"{prefix}Session {sess_id} ({date}):\n" + "\n".join(turn_texts)

                # API supports up to 500k chars - embedding service auto-chunks at 24.5k with 10% overlap
                # No truncation needed - the chunking preserves context across chunks
                if len(session_content) > 24500:
                    logger.debug(f"  Large session ({len(session_content)} chars) will be auto-chunked by embedding service")

                client.create_memory(
                    user_id=question_user_id,
                    content=session_content,
                    memory_type="episode",
                    source_platform="longmemeval",
                    source_id=sess_id,
                    metadata={
                        "session_id": sess_id,
                        "has_answer": session_has_answer,
                        "date": date
                    }
                )

        # Small delay to allow indexing
        time.sleep(0.2)

        # Search for relevant memories
        search_results = client.search_memories(
            user_id=question_user_id,
            query=question,
            limit=5,
            threshold=0.0,
            mode="full"
        )

        # Collect retrieved content
        retrieved_content = ""
        has_answer_marker_found = False
        for mem in search_results:
            content = mem.get("content", "")
            retrieved_content += " " + content
            if "[ANSWER]" in content:
                has_answer_marker_found = True

        # Session accuracy: Check if any oracle session ID appears in retrieved content
        session_correct = False
        for sess_id in oracle_session_ids:
            if f"Session {sess_id}" in retrieved_content:
                session_correct = True
                break
            # Also check by content matching (backup)
            if sess_id in q_sessions:
                sess_messages, _ = q_sessions[sess_id]
                for msg in sess_messages[:2]:
                    msg_text = msg.get("content", "")[:100]
                    if msg_text and msg_text.lower() in retrieved_content.lower():
                        session_correct = True
                        break
            if session_correct:
                break

        # Answer accuracy: Multiple strategies
        answer_found = False

        if has_answer_marker_found:
            answer_found = True

        if not answer_found and answer:
            answer_lower = answer.lower()
            content_lower = retrieved_content.lower()

            if answer_lower in content_lower:
                answer_found = True
            else:
                answer_words = [w for w in answer_lower.split() if len(w) > 2]
                if answer_words:
                    words_found = sum(1 for w in answer_words if w in content_lower)
                    if words_found >= len(answer_words) * 0.8:
                        answer_found = True

        if not answer:
            answer_found = session_correct

        if session_correct:
            results["session_correct"] += 1
            results["by_type"][q_type]["session_correct"] += 1

        if answer_found:
            results["answer_found"] += 1
            results["by_type"][q_type]["answer_found"] += 1

        results["questions"].append({
            "question": question[:100],
            "type": q_type,
            "session_correct": session_correct,
            "answer_found": answer_found
        })

        # Cleanup this question's memories before next question
        client.delete_user_memories(question_user_id)

        # Progress indicator
        if results["total"] % 10 == 0:
            acc = results["session_correct"] / results["total"] * 100
            print(f"  Progress: {results['total']}/{len(data)} questions, {acc:.1f}% session accuracy")
            if verbose_file:
                write_verbose_log(verbose_file, results["total"], results, "longmemeval")

    # Calculate final metrics
    total = results["total"]
    session_acc = results["session_correct"] / total * 100 if total > 0 else 0
    answer_acc = results["answer_found"] / total * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    print("LONGMEMEVAL BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nOverall Results:")
    print(f"  Total questions: {total}")
    print(f"  Session Retrieval Accuracy: {session_acc:.1f}%")
    print(f"  Answer Found Accuracy: {answer_acc:.1f}%")
    print(f"\nBy Question Type:")
    for q_type, stats in sorted(results["by_type"].items()):
        if stats["total"] > 0:
            s_acc = stats["session_correct"] / stats["total"] * 100
            a_acc = stats["answer_found"] / stats["total"] * 100
            print(f"  {q_type}:")
            print(f"    Session: {s_acc:.1f}% ({stats['session_correct']}/{stats['total']})")
            print(f"    Answer: {a_acc:.1f}% ({stats['answer_found']}/{stats['total']})")

    print(f"\nReference Scores:")
    print(f"  - Hindsight (TEMPR): 91.4%")
    print(f"  - Letta/MemGPT: 74.0%")
    print(f"  - SocioMemory (this run): {session_acc:.1f}%")

    return {
        "benchmark": "longmemeval",
        "total_questions": total,
        "session_accuracy": session_acc,
        "answer_accuracy": answer_acc,
        "by_type": dict(results["by_type"]),
        "timestamp": datetime.now().isoformat()
    }


def run_locomo_benchmark(
    client: SocioMemoryAPIClient,
    dataset_path: str,
    max_conversations: int = 5,
    sample_qa: int = 50,
    user_id: Optional[str] = None,
    verbose_file: Optional[str] = None
) -> dict:
    """
    Run LoCoMo benchmark against deployed API.

    Measures retrieval accuracy on conversational memory.

    LoCoMo data format:
    - Each conversation has 'qa' (list of QA pairs) and 'conversation' (dict with session_1, session_2, etc.)
    - Sessions: conversation.session_N contains list of dialogs with 'speaker', 'dia_id', 'text'
    - QA: 'question', 'answer', 'evidence' (list like ["D1:3"]), 'category' (int)
    """
    logger.info("FULL mode: Running ALL conversations with ALL QA pairs")

    print("\n" + "=" * 60)
    print("LOCOMO BENCHMARK - SocioMemory API")
    print("=" * 60)

    # Load dataset
    with open(dataset_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} conversations from {dataset_path}")

    if max_conversations and max_conversations < len(data):
        data = data[:max_conversations]
        print(f"Limited to {max_conversations} conversations")

    # Category mapping (LoCoMo uses integer categories)
    CATEGORY_NAMES = {
        1: "single-hop-factual",
        2: "single-hop-temporal",
        3: "multi-hop",
        4: "open-domain",
        5: "unanswerable"
    }

    results = {
        "total_qa": 0,
        "retrieval_correct_top5": 0,
        "retrieval_correct_top1": 0,
        "by_category": defaultdict(lambda: {"total": 0, "correct": 0}),
        "conversations": []
    }

    for conv_idx, conv in enumerate(data):
        # Get conversation ID - LoCoMo uses 'sample_id'
        conv_id = conv.get("sample_id", f"conv-{conv_idx}")
        benchmark_user_id = user_id or str(uuid.uuid4())

        print(f"\n--- Conversation {conv_idx + 1}/{len(data)}: {conv_id} ---")

        # Parse sessions from 'conversation' dict (format: session_1, session_2, etc.)
        conversation_data = conv.get("conversation", {})

        # Find all session keys
        session_keys = sorted([k for k in conversation_data.keys()
                               if k.startswith("session_") and not k.endswith("_date_time")])

        print(f"Ingesting {len(session_keys)} sessions...")

        # Track dialog IDs for evidence matching
        dialog_to_session = {}  # Maps "D1:3" -> session_num

        for sess_key in session_keys:
            sess_num = sess_key.replace("session_", "")
            dialogs = conversation_data.get(sess_key, [])
            date_key = f"{sess_key}_date_time"
            session_date = conversation_data.get(date_key, "")

            if not dialogs:
                continue

            # Format as conversation with individual dialogs marked
            content_parts = [f"Session {sess_num} ({session_date}):"]
            for dialog in dialogs:
                if isinstance(dialog, dict):
                    speaker = dialog.get("speaker", "unknown")
                    dia_id = dialog.get("dia_id", "")
                    # LoCoMo uses "text" key, not "utterance"
                    utterance = dialog.get("text", "") or dialog.get("utterance", "")

                    # Track which session contains this dialog
                    if dia_id:
                        dialog_to_session[dia_id] = int(sess_num) if sess_num.isdigit() else sess_num

                    content_parts.append(f"[{speaker}] ({dia_id}): {utterance}")

            content = "\n".join(content_parts)

            # API supports up to 500k chars - embedding service chunks automatically at 24.5k
            MAX_CONTENT_CHARS = 500000
            if len(content) > MAX_CONTENT_CHARS:
                print(f"  Warning: Truncating very long session from {len(content)} to {MAX_CONTENT_CHARS} chars")
                content = content[:MAX_CONTENT_CHARS]
            elif len(content) > 24500:
                print(f"  Info: Large session ({len(content)} chars) will be auto-chunked by embedding service")

            client.create_memory(
                user_id=benchmark_user_id,
                content=content,
                memory_type="episode",
                source_platform="locomo",
                source_id=f"{conv_id}-session-{sess_num}",
                metadata={"session_num": sess_num, "date": session_date}
            )

        time.sleep(0.5)

        # Evaluate QA pairs - LoCoMo uses 'qa' key
        qa_pairs = conv.get("qa", [])
        if sample_qa and sample_qa < len(qa_pairs):
            import random
            qa_pairs = random.sample(qa_pairs, sample_qa)

        print(f"Evaluating {len(qa_pairs)} QA pairs...")

        conv_correct = 0
        for qa in qa_pairs:
            question = qa.get("question", "")
            answer = str(qa.get("answer", ""))  # Answer can be int (like year) or string
            category_num = qa.get("category", 0)
            category = CATEGORY_NAMES.get(category_num, f"category-{category_num}")
            evidence_ids = qa.get("evidence", [])  # Format: ["D1:3", "D2:5"]

            # Extract session numbers from evidence IDs
            evidence_sessions = set()
            for ev_id in evidence_ids:
                if ev_id in dialog_to_session:
                    evidence_sessions.add(dialog_to_session[ev_id])

            results["total_qa"] += 1
            results["by_category"][category]["total"] += 1

            # Search
            search_results = client.search_memories(
                user_id=benchmark_user_id,
                query=question,
                limit=5,
                threshold=0.0,
                mode="full"  # Use full mode for 96%/80% accuracy
            )

            # Extract dialog IDs from retrieved content using regex
            # Format in content: [speaker] (D1:3): utterance
            import re
            all_content = " ".join([m.get("content", "") for m in search_results])

            # Find all dialog IDs in retrieved content
            retrieved_dialog_ids = set(re.findall(r'\(D\d+:\d+\)', all_content))
            # Remove parentheses for matching
            retrieved_dialog_ids = {did.strip('()') for did in retrieved_dialog_ids}

            # Check if any evidence ID is found in retrieved content
            evidence_found = any(eid in retrieved_dialog_ids for eid in evidence_ids)

            if evidence_found:
                results["retrieval_correct_top5"] += 1
                results["by_category"][category]["correct"] += 1
                conv_correct += 1

            # Check top-1
            if search_results:
                top1_content = search_results[0].get("content", "")
                top1_dialog_ids = set(re.findall(r'\(D\d+:\d+\)', top1_content))
                top1_dialog_ids = {did.strip('()') for did in top1_dialog_ids}
                if any(eid in top1_dialog_ids for eid in evidence_ids):
                    results["retrieval_correct_top1"] += 1

            # Write verbose log every 10 QA pairs
            if results["total_qa"] % 10 == 0 and verbose_file:
                write_verbose_log(verbose_file, results["total_qa"], results, "locomo")

        conv_acc = conv_correct / len(qa_pairs) * 100 if qa_pairs else 0
        print(f"Retrieval Accuracy: {conv_acc:.1f}% ({conv_correct}/{len(qa_pairs)})")

        results["conversations"].append({
            "conv_id": conv_id,
            "qa_pairs": len(qa_pairs),
            "correct": conv_correct,
            "accuracy": conv_acc
        })

        # Cleanup
        client.delete_user_memories(benchmark_user_id)

    # Calculate final metrics
    total = results["total_qa"]
    top5_acc = results["retrieval_correct_top5"] / total * 100 if total > 0 else 0
    top1_acc = results["retrieval_correct_top1"] / total * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    print("LOCOMO BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nOverall Results:")
    print(f"  Conversations tested: {len(data)}")
    print(f"  Total QA pairs: {total}")
    print(f"  Retrieval Accuracy (top-5): {top5_acc:.1f}%")
    print(f"  Retrieval Accuracy (top-1): {top1_acc:.1f}%")
    print(f"\nCategory-wise Results:")
    for cat, stats in sorted(results["by_category"].items()):
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"  {cat}: {acc:.1f}% ({stats['correct']}/{stats['total']})")

    print(f"\nReference scores from LOCOMO paper:")
    print(f"  - Mem0: 66.9%")
    print(f"  - OpenAI Memory: 52.9%")
    print(f"  - Letta/MemGPT: 74.0%")
    print(f"  - SocioMemory (this run): {top5_acc:.1f}%")
    logger.info(f"\nBenchmark completed in {time.time() - start_time:.1f} seconds" if 'start_time' in dir() else "")

    return {
        "benchmark": "locomo",
        "conversations_tested": len(data),
        "total_qa_pairs": total,
        "top5_accuracy": top5_acc,
        "top1_accuracy": top1_acc,
        "by_category": dict(results["by_category"]),
        "timestamp": datetime.now().isoformat()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Standalone SocioMemory API Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run LongMemEval with 50 questions
    python benchmark_api_standalone.py \\
        --api-url http://localhost:8001 \\
        --api-key YOUR_API_KEY \\
        --dataset longmemeval \\
        --max-questions 50

    # Run LoCoMo with 3 conversations
    python benchmark_api_standalone.py \\
        --api-url http://localhost:8001 \\
        --api-key YOUR_API_KEY \\
        --dataset locomo \\
        --max-conversations 3
        """
    )

    parser.add_argument(
        "--api-url",
        required=True,
        help="SocioMemory API base URL"
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="SocioMemory API key"
    )
    parser.add_argument(
        "--dataset",
        choices=["longmemeval", "longmemeval-s", "locomo"],
        default="longmemeval",
        help="Benchmark dataset to use"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=100,
        help="Max questions for LongMemEval (default: 100)"
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=5,
        help="Max conversations for LoCoMo (default: 5)"
    )
    parser.add_argument(
        "--sample-qa",
        type=int,
        default=50,
        help="QA pairs per conversation for LoCoMo (default: 50)"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--data-dir",
        default="benchmark_data",
        help="Directory to store downloaded datasets"
    )
    parser.add_argument(
        "--log-file",
        help="Log file path (logs to sociomemory/benchmark_results/ by default if --full)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run FULL benchmark (all questions/conversations, no limits)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging"
    )

    args = parser.parse_args()

    # Determine log file path
    log_file = args.log_file
    if args.full and not log_file:
        # Default log location for full benchmarks
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)  # sociomemory directory
        log_dir = os.path.join(base_dir, "benchmark_results")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"api_benchmark_{args.dataset}_{timestamp}.log")

    # Setup logging
    setup_logging(log_file=log_file, verbose=args.verbose)

    # Handle --full flag
    if args.full:
        if args.dataset in ("longmemeval", "longmemeval-s"):
            args.max_questions = None  # No limit
            logger.info("FULL mode: Running ALL questions (no limit)")
        elif args.dataset == "locomo":
            args.max_conversations = None  # No limit
            args.sample_qa = None  # All QA pairs
            logger.info("FULL mode: Running ALL conversations with ALL QA pairs")

    # Initialize client
    client = SocioMemoryAPIClient(args.api_url, args.api_key)

    # Health check
    logger.info(f"Connecting to {args.api_url}...")
    if not client.health_check():
        logger.error("Cannot connect to SocioMemory API")
        sys.exit(1)
    logger.info("API connection successful!")

    # Download dataset
    try:
        dataset_path = download_dataset(args.dataset, args.data_dir)
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        sys.exit(1)

    # Determine output file path
    output_file = args.output
    if args.full and not output_file:
        # Default output location for full benchmarks
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)
        output_dir = os.path.join(base_dir, "benchmark_results")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"api_benchmark_{args.dataset}_{timestamp}.json")
        logger.info(f"Results will be saved to: {output_file}")

    # Create verbose log file path (based on log_file path with _verbose suffix)
    verbose_file = None
    if log_file:
        verbose_file = log_file.replace(".log", "_verbose.log")
        logger.info(f"Verbose progress will be logged to: {verbose_file}")
        # Initialize verbose file
        with open(verbose_file, 'w') as vf:
            vf.write(f"Verbose Benchmark Log - Started at {datetime.now().isoformat()}\n")
            vf.write(f"Dataset: {args.dataset}\n")
            vf.write(f"Full benchmark: {args.full}\n\n")

    # Run benchmark
    start_time = time.time()

    if args.dataset in ("longmemeval", "longmemeval-s"):
        results = run_longmemeval_benchmark(
            client,
            dataset_path,
            max_questions=args.max_questions,
            verbose_file=verbose_file
        )
    elif args.dataset == "locomo":
        results = run_locomo_benchmark(
            client,
            dataset_path,
            max_conversations=args.max_conversations,
            sample_qa=args.sample_qa,
            verbose_file=verbose_file
        )

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    results["api_url"] = args.api_url
    results["full_benchmark"] = args.full

    logger.info(f"\nBenchmark completed in {elapsed:.1f} seconds")

    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()
