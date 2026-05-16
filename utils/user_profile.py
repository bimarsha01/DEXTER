import getpass
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any

from utils.logger import get_logger

logger = get_logger("user_profile")

_DATA_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) / "data"
PROFILE_FILE = _DATA_DIR / "user_profile.json"


@dataclass
class UserProfileData:
    system_username: str = ""
    preferred_name: str = ""
    city: str = ""
    timezone: str = ""
    preferences: Dict[str, Any] = field(default_factory=dict)


class UserProfile:
    """
    Manages the user's persistent profile, automatically detecting the system
    username on first run.
    """

    def __init__(self, config):
        self._config = config
        self.data = UserProfileData()
        self._is_new = False
        self._load()

    def _load(self) -> None:
        """Load user profile from disk, creating it if it doesn't exist."""
        if PROFILE_FILE.exists():
            try:
                raw = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
                self.data = UserProfileData(**raw)
                logger.info("user_profile_loaded", name=self.name)
            except Exception as e:
                logger.warning("user_profile_load_failed", error=str(e))
                self._initialize_new()
        else:
            self._initialize_new()

    def _initialize_new(self) -> None:
        """Auto-detect basics on first run."""
        self._is_new = True
        try:
            sys_user = getpass.getuser()
            self.data.system_username = sys_user
            self.data.preferred_name = sys_user.capitalize()
        except Exception:
            self.data.system_username = "User"
            self.data.preferred_name = "User"

        # Try to seed from config if present
        cfg_city = getattr(self._config.defaults, "city", "")
        if cfg_city:
            self.data.city = cfg_city

        self.save()
        logger.info("user_profile_created", initial_name=self.data.preferred_name)

    def save(self) -> None:
        """Persist profile to disk."""
        PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            PROFILE_FILE.write_text(json.dumps(asdict(self.data), indent=2), encoding="utf-8")
        except Exception as e:
            logger.error("user_profile_save_failed", error=str(e))

    @property
    def name(self) -> str:
        """Returns the user's preferred name."""
        return self.data.preferred_name or self.data.system_username

    @property
    def city(self) -> str:
        """Returns the user's city, falling back to config default."""
        return self.data.city or getattr(self._config.defaults, "city", "")

    def is_first_run(self) -> bool:
        """Returns True if the profile was just created."""
        return self._is_new

    def update_preference(self, key: str, value: Any) -> None:
        """Update an arbitrary user preference and save."""
        self.data.preferences[key] = value
        self.save()
        logger.info("user_preference_updated", key=key)
