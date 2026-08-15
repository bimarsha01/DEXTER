"""
Discovers known proper nouns from the user's environment.
Runs at startup in a background thread to keep vocabulary fresh.
Sources: installed apps, indexed document filenames, folder names,
config values (user city).
"""

import os
import re
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("vocabulary_builder")


class VocabularyBuilder:
    """Discovers and registers vocabulary terms from the user's environment."""

    def __init__(self, asr_engine, config) -> None:
        self._asr = asr_engine
        self._config = config

    def build_all(self) -> int:
        """
        Discover and register all vocabulary sources.
        Returns total terms added.
        Called at startup in a background thread.
        """
        terms: list[str] = []
        terms += self._get_installed_apps()
        terms += self._get_indexed_filenames()
        terms += self._get_user_folder_names()
        terms += self._get_city_from_config()

        # Deduplicate
        unique = list(set(t for t in terms if t and len(t) >= 3))
        if unique:
            self._asr.add_vocabulary(unique)
        logger.info(
            "vocabulary_built",
            total_terms=len(unique),
            sources=["apps", "files", "folders", "config"],
        )
        return len(unique)

    def _get_installed_apps(self) -> list[str]:
        """Read installed application names from Windows registry."""
        apps: list[str] = []
        try:
            import winreg

            registry_paths = [
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
            ]
            for reg_path in registry_paths:
                try:
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path)
                    for i in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            sub_key = winreg.EnumKey(key, i)
                            sub = winreg.OpenKey(key, sub_key)
                            name, _ = winreg.QueryValueEx(sub, "DisplayName")
                            if name and isinstance(name, str):
                                apps.append(name)
                                apps.extend(name.split())
                        except (FileNotFoundError, OSError):
                            continue
                except Exception as e:
                    logger.debug("registry_read_failed", error=str(e))
        except ImportError:
            logger.debug("winreg_not_available")
        return apps

    def _get_indexed_filenames(self) -> list[str]:
        """Extract filenames from RAG index as vocabulary."""
        names: list[str] = []
        try:
            persist_dir = getattr(self._config.rag, "persist_directory", None)
            if not persist_dir:
                return names
            persist_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(str(persist_dir))))
            if not os.path.exists(persist_dir):
                return names
            from utils.chroma_client import get_persistent_client

            client = get_persistent_client(persist_dir)
            collections = client.list_collections()
            for col in collections:
                try:
                    if isinstance(col, str):
                        col = client.get_collection(col)
                    elif not hasattr(col, "get") and hasattr(col, "name"):
                        col = client.get_collection(col.name)
                    results = col.get(include=["metadatas"], limit=5000)
                    for meta in results.get("metadatas", []):
                        if meta and "path" in meta:
                            fname = Path(meta["path"]).stem
                            names.append(fname)
                            # Split CamelCase and hyphenated names
                            parts = re.findall(
                                r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?=[A-Z]|$)",
                                fname,
                            )
                            names.extend(parts)
                except Exception:
                    continue
        except Exception as e:
            logger.debug("rag_vocabulary_failed", error=str(e))
        return names

    def _get_user_folder_names(self) -> list[str]:
        """Read folder names from configured watched directories."""
        names: list[str] = []
        watched = getattr(self._config.rag, "personal_roots", [])
        for directory in watched:
            try:
                expanded = os.path.expandvars(os.path.expanduser(str(directory)))
                p = Path(expanded)
                if p.exists() and p.is_dir():
                    for item in p.iterdir():
                        if item.is_dir() and not item.name.startswith("."):
                            names.append(item.name)
            except Exception:
                continue
        return names

    def _get_city_from_config(self) -> list[str]:
        """Add the user's configured city to vocabulary."""
        city = getattr(self._config.defaults, "city", None)
        if city and isinstance(city, str) and city.strip():
            return [city.strip()]
        return []
