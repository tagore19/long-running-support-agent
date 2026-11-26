"""Layer 4 hybrid semantic (TF-IDF/lexical) and exact entity retrieval in pure Python/SQLite."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from src.database.models import EpisodeRecord
from src.database.repositories import SupportRepository
from src.memory.memory_schemas import (
    RetrievedMemoryBundle,
    RetrievedMemoryItem,
    extract_entities,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "my",
        "me",
        "i",
        "you",
        "your",
        "it",
        "this",
        "that",
        "can",
        "please",
        "what",
        "did",
        "do",
    }
)


def _tokenize(text: str) -> list[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _tfidf_cosine(query_tokens: list[str], doc_tokens_list: list[list[str]]) -> list[float]:
    """Compute TF-IDF cosine similarity between query and candidate documents."""
    if not query_tokens or not doc_tokens_list:
        return [0.0] * len(doc_tokens_list)

    n_docs = len(doc_tokens_list) + 1
    df: Counter[str] = Counter(set(query_tokens))
    for doc_tokens in doc_tokens_list:
        df.update(set(doc_tokens))

    def vectorize(tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)
        total = max(1, len(tokens))
        return {
            term: (count / total) * (math.log((n_docs + 1) / (df[term] + 1)) + 1.0)
            for term, count in tf.items()
        }

    q_vec = vectorize(query_tokens)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

    scores: list[float] = []
    for doc_tokens in doc_tokens_list:
        d_vec = vectorize(doc_tokens)
        d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
        dot = sum(q_vec[t] * d_vec.get(t, 0.0) for t in q_vec)
        scores.append(dot / (q_norm * d_norm))
    return scores


class HybridMemoryRetriever:
    """Pure-Python hybrid retriever combining exact ID lookup, SQL filters, lexical similarity, importance, and recency."""

    def __init__(self, repo: SupportRepository) -> None:
        self.repo = repo

    def lookup_exact_records(
        self, *, customer_id: str, entities: list[str]
    ) -> list[dict[str, Any]]:
        """Look up authoritative Layer 1/2 SQLite records matching extracted entity IDs, scoped to customer_id."""
        exact: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        customer_refunds = self.repo.get_refunds_for_customer(customer_id)
        for entity in entities:
            if entity.startswith("ORD-") and entity not in seen_ids:
                order = self.repo.get_order(entity)
                if order is not None and order.customer_id == customer_id:
                    seen_ids.add(entity)
                    exact.append({"record_type": "order", "record": order.model_dump()})
                    for r in self.repo.get_refunds_for_order(entity):
                        if r.refund_id not in seen_ids:
                            seen_ids.add(r.refund_id)
                            exact.append({"record_type": "refund", "record": r.model_dump()})
                    for evt in self.repo.list_events(
                        customer_id=customer_id, order_id=entity, limit=10
                    ):
                        if evt.event_id not in seen_ids:
                            seen_ids.add(evt.event_id)
                            exact.append({"record_type": "event", "record": evt.model_dump()})
            elif entity.startswith("TKT-") and entity not in seen_ids:
                ticket = self.repo.get_support_ticket(entity)
                if ticket is not None and ticket.customer_id == customer_id:
                    seen_ids.add(entity)
                    exact.append({"record_type": "support_ticket", "record": ticket.model_dump()})
            elif entity.startswith("REF-") and entity not in seen_ids:
                for r in customer_refunds:
                    if r.refund_id == entity:
                        seen_ids.add(entity)
                        exact.append({"record_type": "refund", "record": r.model_dump()})
        return exact

    def retrieve(
        self,
        *,
        customer_id: str,
        thread_id: str,
        query: str,
        top_k: int = 5,
        min_importance: float = 0.0,
        include_cross_thread_episodes: bool = True,
    ) -> list[RetrievedMemoryItem]:
        """Retrieve ranked memory items scoped strictly to customer_id."""
        query_entities = set(extract_entities(query))
        query_tokens = _tokenize(query)

        episodes = self.repo.list_episodes(
            customer_id=customer_id,
            thread_id=None if include_cross_thread_episodes else thread_id,
            min_importance=min_importance,
            limit=100,
        )
        turns = self.repo.list_turns(
            thread_id=thread_id,
            customer_id=customer_id,
            limit=200,
        )
        events = self.repo.list_events(
            customer_id=customer_id,
            thread_id=thread_id,
            limit=100,
        )

        candidates: list[dict[str, object]] = []
        for idx, ep in enumerate(episodes):
            thread_bonus = 0.05 if ep.thread_id == thread_id else 0.0
            candidates.append(
                {
                    "source": "episode",
                    "item_id": ep.episode_id,
                    "content": ep.content,
                    "entities": ep.entities,
                    "importance": min(1.0, ep.importance + thread_bonus),
                    "recency": 1.0 - (idx / max(1, len(episodes))) * 0.4,
                    "created_at": ep.created_at,
                }
            )

        total_turns = max(1, len(turns))
        for idx, turn in enumerate(turns):
            if turn.role == "tool":
                continue
            candidates.append(
                {
                    "source": "turn",
                    "item_id": f"TURN-{turn.turn_index}-{turn.role}",
                    "content": f"[{turn.role} turn {turn.turn_index}] {turn.content}",
                    "entities": turn.entities,
                    "importance": 0.65 if turn.entities else 0.45,
                    "recency": 0.4 + 0.6 * ((idx + 1) / total_turns),
                    "created_at": turn.created_at,
                }
            )

        for idx, evt in enumerate(events):
            evt_entities = [e for e in [evt.order_id, evt.customer_id] if e]
            summary = (
                f"Event {evt.event_type} for order {evt.order_id or 'N/A'} "
                f"(amount={evt.amount}, actor={evt.actor})"
            )
            candidates.append(
                {
                    "source": "event",
                    "item_id": evt.event_id,
                    "content": summary,
                    "entities": evt_entities,
                    "importance": 0.85,
                    "recency": 0.5 + 0.5 * ((idx + 1) / max(1, len(events))),
                    "created_at": evt.timestamp,
                }
            )

        if not candidates:
            return []

        doc_tokens_list = [_tokenize(str(c["content"])) for c in candidates]
        lexical_scores = _tfidf_cosine(query_tokens, doc_tokens_list)

        ranked: list[RetrievedMemoryItem] = []
        seen_contents: set[str] = set()
        for c, lex_score in zip(candidates, lexical_scores, strict=True):
            c_entities = set(c["entities"])  # type: ignore[arg-type]
            entity_overlap = len(query_entities & c_entities)
            exact_match = entity_overlap > 0
            entity_score = min(1.0, entity_overlap * 0.6) if exact_match else 0.0

            combined = (
                0.42 * entity_score
                + 0.30 * lex_score
                + 0.16 * float(c["importance"])  # type: ignore[arg-type]
                + 0.12 * float(c["recency"])  # type: ignore[arg-type]
            )
            norm_key = str(c["content"]).strip().lower()
            if norm_key in seen_contents:
                continue
            seen_contents.add(norm_key)
            ranked.append(
                RetrievedMemoryItem(
                    source=c["source"],  # type: ignore[arg-type]
                    item_id=str(c["item_id"]),
                    content=str(c["content"]),
                    entities=list(c_entities),
                    score=round(combined, 4),
                    exact_entity_match=exact_match,
                    created_at=str(c["created_at"]),
                )
            )

        ranked.sort(key=lambda item: (item.exact_entity_match, item.score), reverse=True)
        return ranked[:top_k]

    def retrieve_bundle(
        self,
        *,
        customer_id: str,
        thread_id: str,
        query: str,
        recent_messages: list[dict[str, str]] | None = None,
        top_k: int = 5,
        recent_window: int = 10,
        min_importance: float = 0.0,
    ) -> RetrievedMemoryBundle:
        """Return a complete Layer 4 RetrievedMemoryBundle across exact, episodic, and historical records."""
        entities = extract_entities(query)
        exact_records = self.lookup_exact_records(customer_id=customer_id, entities=entities)

        # Rank customer episodes across threads by query relevance + importance + recency
        all_episodes = self.repo.list_episodes(
            customer_id=customer_id,
            min_importance=min_importance,
            limit=50,
        )
        ranked_episodes = self._rank_episodes(
            episodes=all_episodes,
            query=query,
            query_entities=set(entities),
            thread_id=thread_id,
        )

        relevant_items = self.retrieve(
            customer_id=customer_id,
            thread_id=thread_id,
            query=query,
            top_k=top_k,
            min_importance=min_importance,
        )
        preserved_recent = list((recent_messages or [])[-recent_window:])

        return RetrievedMemoryBundle(
            exact_records=exact_records,
            episodes=ranked_episodes[:top_k],
            relevant_messages=relevant_items,
            recent_messages=preserved_recent,
        )

    @staticmethod
    def _rank_episodes(
        *,
        episodes: list[EpisodeRecord],
        query: str,
        query_entities: set[str],
        thread_id: str,
    ) -> list[EpisodeRecord]:
        if not episodes:
            return []
        q_tokens = _tokenize(query)
        doc_tokens = [_tokenize(ep.content) for ep in episodes]
        lex_scores = _tfidf_cosine(q_tokens, doc_tokens)

        scored: list[tuple[float, EpisodeRecord]] = []
        total = max(1, len(episodes))
        for idx, (ep, lex) in enumerate(zip(episodes, lex_scores, strict=True)):
            ep_entities = set(ep.entities)
            exact_hit = 1.0 if (query_entities & ep_entities) else 0.0
            recency = 1.0 - (idx / total) * 0.3
            thread_bonus = 0.05 if ep.thread_id == thread_id else 0.0
            score = 0.38 * exact_hit + 0.30 * lex + 0.22 * ep.importance + 0.10 * recency + thread_bonus
            scored.append((score, ep))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ep for _, ep in scored]
