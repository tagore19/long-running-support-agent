"""Layer 3 episodic memory extraction for preferences, commitments, and issues."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.config import Settings
from src.database.models import EpisodeRecord
from src.database.repositories import SupportRepository
from src.memory.memory_schemas import extract_entities

PREFERENCE_PATTERNS = [
    re.compile(r"\bprefer(?:s|red)?\b", re.IGNORECASE),
    re.compile(r"\bnever\s+(?:call|sms|text|spam)\b", re.IGNORECASE),
    re.compile(r"\bonly\s+email\b", re.IGNORECASE),
    re.compile(r"\bemail\s+updates\b", re.IGNORECASE),
    re.compile(r"\bmoving\b.*\bapartment\b", re.IGNORECASE),
    re.compile(r"\bpreferred\s+(?:contact|language)\b", re.IGNORECASE),
]

ISSUE_PATTERNS = [
    re.compile(
        r"\b(?:broken|damaged|leaking|chipped|dented|defective|complaint|frustrated|logged\s+out|weak)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:escalat|human\s+agent|specialist|supervisor)\b", re.IGNORECASE),
]

COMMITMENT_PATTERNS = [
    re.compile(r"\b(?:refund|ticket|escalat|created|issued|approved|follow\s+up)\b", re.IGNORECASE),
]


class ExtractedEpisodeDraft(BaseModel):
    """Schema-validated candidate episode before persistence."""

    content: str
    topic: str
    importance: float = Field(ge=0.0, le=1.0)
    entities: list[str] = Field(default_factory=list)


class ExtractedEpisodeList(BaseModel):
    """Structured LLM extraction container."""

    episodes: list[ExtractedEpisodeDraft] = Field(default_factory=list)


class EpisodeExtractor:
    """Extracts non-authoritative Layer 3 episodes from conversation turns."""

    def __init__(self, repo: SupportRepository, settings: Settings | None = None) -> None:
        self.repo = repo
        self.settings = settings

    def extract(
        self,
        *,
        customer_id: str,
        thread_id: str,
        user_message: str,
        assistant_message: str,
        use_llm: bool = False,
    ) -> list[EpisodeRecord]:
        """Extract structured Layer 3 episodes deterministically (with optional LLM assist)."""
        drafts = self._deterministic_extract(user_message, assistant_message)

        if use_llm and self.settings and self.settings.effective_llm_provider == "gemini":
            llm_drafts = self._llm_extract(user_message, assistant_message)
            existing_contents = {d.content.lower() for d in drafts}
            for d in llm_drafts:
                if d.content.lower() not in existing_contents:
                    drafts.append(d)

        created: list[EpisodeRecord] = []
        for draft in drafts:
            created.append(
                self.repo.add_episode(
                    customer_id=customer_id,
                    thread_id=thread_id,
                    content=draft.content,
                    entities=draft.entities,
                    topic=draft.topic,
                    importance=draft.importance,
                )
            )
        return created

    @staticmethod
    def _deterministic_extract(
        user_message: str, assistant_message: str
    ) -> list[ExtractedEpisodeDraft]:
        drafts: list[ExtractedEpisodeDraft] = []
        user_entities = extract_entities(user_message)
        assistant_entities = extract_entities(assistant_message)
        combined_entities = list(dict.fromkeys(user_entities + assistant_entities))

        if any(p.search(user_message) for p in PREFERENCE_PATTERNS):
            drafts.append(
                ExtractedEpisodeDraft(
                    content=f"Customer preference stated (advisory): {user_message.strip()}",
                    entities=user_entities,
                    topic="preference",
                    importance=0.92,
                )
            )

        if any(p.search(user_message) for p in ISSUE_PATTERNS):
            drafts.append(
                ExtractedEpisodeDraft(
                    content=f"Customer reported issue (advisory): {user_message.strip()}",
                    entities=combined_entities,
                    topic="issue",
                    importance=0.78,
                )
            )

        if assistant_entities and any(p.search(assistant_message) for p in COMMITMENT_PATTERNS):
            snippet = assistant_message.strip()
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."
            drafts.append(
                ExtractedEpisodeDraft(
                    content=f"Agent response/commitment (advisory): {snippet}",
                    entities=combined_entities,
                    topic="commitment",
                    importance=0.86,
                )
            )
        return drafts

    def _llm_extract(
        self, user_message: str, assistant_message: str
    ) -> list[ExtractedEpisodeDraft]:
        if not self.settings or not self.settings.gemini_api_key:
            return []
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.settings.gemini_api_key)
            prompt = (
                "Extract conversational preferences, unresolved complaints, or commitments as JSON. "
                "Do NOT fabricate financial approvals.\n"
                f"User: {user_message}\nAssistant: {assistant_message}"
            )
            resp = client.models.generate_content(
                model=self.settings.compaction_model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=ExtractedEpisodeList,
                ),
            )
            parsed = ExtractedEpisodeList.model_validate_json(resp.text)
            return parsed.episodes
        except Exception:
            return []


def extract_and_store_episodes(
    *,
    repo: SupportRepository,
    customer_id: str,
    thread_id: str,
    user_message: str,
    assistant_message: str,
) -> list[EpisodeRecord]:
    """Convenience helper delegating to EpisodeExtractor."""
    return EpisodeExtractor(repo).extract(
        customer_id=customer_id,
        thread_id=thread_id,
        user_message=user_message,
        assistant_message=assistant_message,
    )
