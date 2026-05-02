import os
from dataclasses import dataclass
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from utils.logger import get_logger

logger = get_logger("intent_rag")
from utils.config import DexterConfig


@dataclass
class IntentExample:
    intent: str
    tool_name: str
    example: str


@dataclass
class IntentMatch:
    intent: str
    tool_name: str
    score: float
    example: str


class IntentRAG:
    def __init__(self, config: DexterConfig, persist_directory: str = "./memory_db"):
        rag = config.rag
        self.catalog_path = rag.intent_catalog_path
        self.top_k = rag.intent_top_k
        self.min_score = rag.intent_min_score
        self._last_indexed = 0.0

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection_name = "dexter_intents"
        self.collection = self._get_collection()
        self._index_if_needed()

    def match(self, text: str) -> Optional[IntentMatch]:
        if not text or not text.strip():
            return None

        self._index_if_needed()

        results = self.collection.query(
            query_texts=[text],
            n_results=self.top_k,
        )

        if not results.get("documents") or not results["documents"][0]:
            return None

        best_doc = results["documents"][0][0]
        best_meta = results["metadatas"][0][0]
        best_dist = results.get("distances", [[1.0]])[0][0]
        score = 1.0 / (1.0 + float(best_dist))

        if score < self.min_score:
            return None

        return IntentMatch(
            intent=best_meta.get("intent", ""),
            tool_name=best_meta.get("tool", ""),
            score=score,
            example=best_doc,
        )

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def _index_if_needed(self) -> None:
        if not os.path.exists(self.catalog_path):
            return

        mtime = os.path.getmtime(self.catalog_path)
        if mtime <= self._last_indexed:
            return

        self._reindex()
        self._last_indexed = mtime

    def _reindex(self) -> None:
        examples = self._parse_catalog()
        if not examples:
            return

        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            logger.warning("intent_rag_collection_delete_failed", error=str(e), exc_info=True)

        self.collection = self._get_collection()

        ids = []
        documents = []
        metadatas = []

        for index, entry in enumerate(examples):
            ids.append(f"{entry.intent}-{index}")
            documents.append(entry.example)
            metadatas.append({"intent": entry.intent, "tool": entry.tool_name})

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info("intent_rag_indexed", example_count=len(ids))

    def _parse_catalog(self) -> List[IntentExample]:
        examples = []
        current_intent = ""
        current_tool = ""

        try:
            with open(self.catalog_path, "r", encoding="utf-8") as file:
                lines = file.readlines()
        except Exception as e:
            logger.error("intent_catalog_read_failed", error=str(e), exc_info=True)
            return []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if line.lower().startswith("## intent:"):
                current_intent = line.split(":", 1)[1].strip()
                current_tool = ""
                continue

            if line.lower().startswith("tool:"):
                current_tool = line.split(":", 1)[1].strip()
                continue

            if line.startswith("-") and current_intent and current_tool:
                example = line.lstrip("- ").strip()
                if example:
                    examples.append(IntentExample(current_intent, current_tool, example))

        return examples
