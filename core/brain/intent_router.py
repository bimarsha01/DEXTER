import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from utils.config import DexterConfig
from utils.logger import get_logger
from tools.pc_controls import APP_MAP

logger = get_logger("intent_router")


@dataclass
class IntentDecision:
    action: str  # none | tool | ask | vision | cancel
    tool_name: str = ""
    args: Dict = field(default_factory=dict)
    prompt: str = ""
    vision_mode: str = ""  # screen | file
    file_path: str = ""
    requires_confirmation: bool = False


@dataclass
class PendingAction:
    kind: str  # confirm | slot | open_choice
    tool_name: str
    args: Dict
    prompt: str
    expires_at: float


class IntentRouter:
    _CITY_ASR_ALIASES = {
        "cut mondo": "Kathmandu",
        "cut mondo right": "Kathmandu",
        "kat mondo": "Kathmandu",
        "kath mandoo": "Kathmandu",
        "kathmandhu": "Kathmandu",
    }

    def __init__(self, config: DexterConfig):
        self.default_city = (config.defaults.city or "Kathmandu").strip()
        logger.info("intent_router_initialized", has_default_city=bool(self.default_city))

    def resolve_pending(self, text: str, pending: PendingAction) -> IntentDecision:
        lowered = text.lower().strip()
        if lowered in {"cancel", "never mind", "nevermind", "stop"}:
            return IntentDecision(action="cancel", prompt="Understood, sir. I have cancelled that request.")

        if pending.kind == "confirm":
            if lowered in {"yes", "confirm", "do it", "proceed", "ok", "okay"}:
                return IntentDecision(action="tool", tool_name=pending.tool_name, args=pending.args)
            return IntentDecision(action="ask", prompt=pending.prompt)

        if pending.kind == "slot":
            # Slot filling based on tool type
            if pending.tool_name == "get_weather":
                city = self._extract_city(text) or text.strip()
                if city:
                    args = dict(pending.args)
                    args["city"] = city
                    return IntentDecision(action="tool", tool_name=pending.tool_name, args=args)
                return IntentDecision(action="ask", prompt=pending.prompt)

            if pending.tool_name == "vision_file":
                path = text.strip()
                if path:
                    return IntentDecision(action="vision", vision_mode="file", file_path=path)
                return IntentDecision(action="ask", prompt=pending.prompt)

            if pending.tool_name == "open_application":
                app = text.strip()
                if app:
                    args = dict(pending.args)
                    args["app_name"] = app
                    return IntentDecision(action="tool", tool_name=pending.tool_name, args=args)
                return IntentDecision(action="ask", prompt=pending.prompt)

            if pending.tool_name == "copy_to_clipboard":
                content = text.strip()
                if content:
                    args = dict(pending.args)
                    args["text"] = content
                    return IntentDecision(action="tool", tool_name=pending.tool_name, args=args)
                return IntentDecision(action="ask", prompt=pending.prompt)

        if pending.kind == "open_choice":
            match_id = pending.args.get("match_id")
            if match_id:
                return IntentDecision(
                    action="tool",
                    tool_name="resolve_open_target",
                    args={"match_id": match_id, "selection": text.strip()},
                )
            return IntentDecision(action="ask", prompt=pending.prompt)

        return IntentDecision(action="ask", prompt=pending.prompt)

    def detect_intent(self, text: str) -> IntentDecision:
        lowered = text.lower().strip()
        normalized = self._strip_filler_prefixes(lowered)

        direct_app_match = re.match(r"^(?:open|launch|start)\s+(.+)$", normalized)
        if direct_app_match:
            app_name = direct_app_match.group(1).strip()
            if self._should_launch_directly(app_name):
                return IntentDecision(action="tool", tool_name="open_application", args={"app_name": app_name})

        browser_match = re.match(
            r"open\s+(.+?)\s+in\s+(chrome|google chrome|edge|microsoft edge|firefox|brave)$",
            normalized,
        )
        if browser_match:
            target = browser_match.group(1).strip()
            browser = browser_match.group(2).strip()
            url = self._resolve_known_url(target)
            if url:
                return IntentDecision(
                    action="tool",
                    tool_name="open_url_in_browser",
                    args={"url": url, "browser": browser},
                )
            return IntentDecision(
                action="tool",
                tool_name="search_google",
                args={"query": target},
            )

        bare_browser_match = re.match(
            r"^(.+?)\s+(?:in|on)\s+(chrome|google chrome|edge|microsoft edge|firefox|brave)$",
            normalized,
        )
        if bare_browser_match:
            target = bare_browser_match.group(1).strip()
            browser = bare_browser_match.group(2).strip()
            url = self._resolve_known_url(target)
            if url:
                return IntentDecision(
                    action="tool",
                    tool_name="open_url_in_browser",
                    args={"url": url, "browser": browser},
                )
            if self._looks_like_content_request(target):
                return IntentDecision(
                    action="tool",
                    tool_name="search_content_platform",
                    args={
                        "query": target,
                        "platform": browser,
                        "content_type": self._infer_content_type("open", target, browser),
                    },
                )

        content_request = self._detect_content_request(normalized)
        if content_request:
            action, query, platform = content_request
            content_type = self._infer_content_type(action, query, platform)
            if not platform:
                platform = self._default_platform_for_content(content_type)
            return IntentDecision(
                action="tool",
                tool_name="search_content_platform",
                args={
                    "query": query,
                    "platform": platform,
                    "content_type": content_type,
                },
            )

        if self._is_temperature_request(normalized):
            city = self._extract_city(text)
            if not city and self.default_city:
                city = self.default_city
            if not city:
                return IntentDecision(
                    action="ask",
                    tool_name="get_weather",
                    prompt="Which city should I check, sir?",
                )
            return IntentDecision(action="tool", tool_name="get_weather", args={"city": city})

        if self._is_time_request(normalized):
            city = self._extract_city(text)
            return IntentDecision(action="tool", tool_name="get_current_time", args={"city": city})

        # Screenshot tool intents
        if (
            "screenshot" in normalized
            or "screen shot" in normalized
            or "screen capture" in normalized
            or (
                "screen" in normalized
                and any(kw in normalized for kw in ["take", "capture", "grab", "save"])
            )
        ):
            return IntentDecision(action="tool", tool_name="take_screenshot", args={})

        # Vision intents
        if any(
            kw in normalized
            for kw in [
                "look at",
                "see",
                "analyze",
                "what's on my screen",
                "what is on my screen",
            ]
        ):
            return IntentDecision(action="vision", vision_mode="screen")

        file_match = re.search(r"([\w\-./\\]+\.(py|txt|md|json|yaml|yml))", text, re.IGNORECASE)
        if file_match:
            return IntentDecision(action="vision", vision_mode="file", file_path=file_match.group(1))
        if any(kw in lowered for kw in ["function", "code", "bug", "error in this file"]):
            return IntentDecision(action="ask", vision_mode="file", prompt="Which file should I inspect, sir? Provide a relative path.")

        # Weather
        if "weather" in normalized or "forecast" in normalized:
            city = self._extract_city(text)
            if not city:
                if self.default_city:
                    city = self.default_city
                else:
                    return IntentDecision(
                        action="ask",
                        tool_name="get_weather",
                        prompt="Which city should I check, sir?",
                    )
            return IntentDecision(action="tool", tool_name="get_weather", args={"city": city})

        # App launch
        app_match = re.match(r"(?:open|launch|start)\s+(.+)$", normalized)
        if app_match:
            app_name = app_match.group(1).strip()
            if app_name:
                if self._should_launch_directly(app_name):
                    return IntentDecision(action="tool", tool_name="open_application", args={"app_name": app_name})
                return IntentDecision(action="tool", tool_name="resolve_open_target", args={"query": app_name})
            return IntentDecision(action="ask", tool_name="resolve_open_target", prompt="What should I open, sir?")

        # Clipboard
        if "clipboard" in normalized and any(
            kw in normalized for kw in ["read", "show", "what's on", "what is on", "check"]
        ):
            return IntentDecision(action="tool", tool_name="read_clipboard", args={})

        copy_match = re.match(r"(?:copy|set clipboard)\s+(.+)$", text, re.IGNORECASE)
        if copy_match:
            content = copy_match.group(1).strip()
            if content:
                return IntentDecision(action="tool", tool_name="copy_to_clipboard", args={"text": content})
            return IntentDecision(action="ask", tool_name="copy_to_clipboard", prompt="What should I copy to the clipboard, sir?")

        return IntentDecision(action="none")

    def _strip_filler_prefixes(self, text: str) -> str:
        cleaned = re.sub(r"[?.!,]+$", "", (text or "").strip().lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return cleaned
        cleaned = re.sub(
            r"^(?:\b(?:ok|okay|hey|hi|hello|dexter|please)\b[,\s]*)+",
            "",
            cleaned,
        ).strip()
        return cleaned

    def _resolve_known_url(self, target: str) -> str:
        clean = target.strip().lower()
        if not clean:
            return ""
        if "." in clean:
            return clean
        mapping = {
            "youtube": "https://www.youtube.com",
            "youtube music": "https://music.youtube.com",
            "spotify": "https://open.spotify.com",
            "soundcloud": "https://soundcloud.com",
            "netflix": "https://www.netflix.com",
            "prime video": "https://www.primevideo.com",
            "apple music": "https://music.apple.com",
            "espn": "https://www.espn.com",
            "gmail": "https://mail.google.com",
            "google": "https://www.google.com",
        }
        return mapping.get(clean, "")

    def _should_launch_directly(self, app_name: str) -> bool:
        clean = app_name.strip().lower()
        if not clean:
            return False
        if clean in APP_MAP:
            return True
        direct_keywords = {
            "chrome",
            "google chrome",
            "edge",
            "microsoft edge",
            "firefox",
            "brave",
            "spotify",
            "discord",
            "word",
            "excel",
            "powerpoint",
            "outlook",
            "notepad",
            "calculator",
            "file explorer",
            "explorer",
            "vscode",
            "visual studio code",
            "vs code",
        }
        return clean in direct_keywords

    def _detect_content_request(self, normalized: str) -> tuple[str, str, str] | None:
        platform_match = re.match(
            r"^(?:(play|watch|find|search|open)(?:\s+for)?\s+)?(.+?)\s+(?:on|in|from)\s+(.+)$",
            normalized,
        )
        if platform_match:
            action = (platform_match.group(1) or "open").strip()
            query = platform_match.group(2).strip()
            platform = platform_match.group(3).strip()
            if not self._looks_like_content_request(query) and not self._is_known_platform(platform):
                return None
            return action, query, platform

        simple_match = re.match(r"^(play|watch|find|open)(?:\s+for)?\s+(.+)$", normalized)
        if simple_match:
            action = simple_match.group(1).strip()
            query = simple_match.group(2).strip()
            if action == "open" and not self._looks_like_content_request(query):
                return None
            return action, query, ""

        return None

    def _is_known_platform(self, platform: str) -> bool:
        clean = self._normalize_platform(platform)
        known = {
            "youtube",
            "youtube music",
            "spotify",
            "soundcloud",
            "apple music",
            "netflix",
            "prime video",
            "espn",
            "twitch",
        }
        return clean in known

    def _normalize_platform(self, platform: str) -> str:
        cleaned = (platform or "").strip().lower()
        cleaned = cleaned.replace("https://", "").replace("http://", "")
        cleaned = cleaned.split("/")[0]
        cleaned = cleaned.replace("www.", "")
        cleaned = re.sub(r"\s+", " ", cleaned)
        aliases = {
            "yt music": "youtube music",
            "music.youtube": "youtube music",
            "music.apple": "apple music",
            "amazon prime video": "prime video",
        }
        return aliases.get(cleaned, cleaned)

    def _looks_like_content_request(self, query: str) -> bool:
        text = query.lower()
        keywords = [
            "song",
            "music",
            "playlist",
            "album",
            "track",
            "artist",
            "podcast",
            "episode",
            "movie",
            "film",
            "show",
            "series",
            "tv",
            "video",
            "highlight",
            "highlights",
            "sports",
            "sport",
            "match",
            "game",
            "stream",
        ]
        return any(keyword in text for keyword in keywords)

    def _is_temperature_request(self, normalized: str) -> bool:
        return bool(re.search(r"\b(temp|temperature|weather|forecast)\b", normalized))

    def _is_time_request(self, normalized: str) -> bool:
        if not re.search(r"\btime\b", normalized):
            return False
        if any(word in normalized for word in ["timer", "timesheet"]):
            return False
        return True

    def _infer_content_type(self, action: str, query: str, platform: str = "") -> str:
        text = f"{action} {query} {platform}".lower()
        if any(keyword in text for keyword in ["netflix", "prime video", "apple tv", "disney+", "hulu"]):
            if any(keyword in text for keyword in ["show", "series", "episode", "tv"]):
                return "tv"
            return "movie"
        if any(keyword in text for keyword in ["youtube music", "spotify", "soundcloud", "apple music"]):
            return "music"
        if any(keyword in text for keyword in ["podcast", "episode", "interview"]):
            return "podcast"
        if any(keyword in text for keyword in ["movie", "film", "series", "show", "tv"]):
            return "movie"
        if any(keyword in text for keyword in ["highlight", "highlights", "sports", "sport", "match", "game", "f1", "nba", "nfl", "ucl"]):
            return "sports"
        if any(keyword in text for keyword in ["song", "music", "playlist", "album", "track", "artist", "lo-fi", "lofi"]):
            return "music"
        if action == "watch":
            return "video"
        if action == "play":
            return "music"
        return "general"

    def _default_platform_for_content(self, content_type: str) -> str:
        defaults = {
            "music": "youtube music",
            "podcast": "spotify",
            "video": "youtube",
            "movie": "netflix",
            "sports": "youtube",
            "general": "youtube",
        }
        return defaults.get(content_type, "youtube")

    def build_pending_slot(self, decision: IntentDecision, ttl_seconds: int = 45) -> PendingAction:
        return PendingAction(
            kind="slot",
            tool_name=decision.tool_name,
            args=decision.args,
            prompt=decision.prompt,
            expires_at=time.time() + ttl_seconds,
        )

    def build_pending_confirm(self, tool_name: str, args: Dict, prompt: str, ttl_seconds: int = 30) -> PendingAction:
        return PendingAction(
            kind="confirm",
            tool_name=tool_name,
            args=args,
            prompt=prompt,
            expires_at=time.time() + ttl_seconds,
        )

    def _extract_city(self, text: str) -> str:
        cleaned = re.sub(r"[?.!,]+$", "", text.strip())
        cleaned = re.sub(r"\s+", " ", cleaned)

        def _sanitize_city(value: str) -> str:
            city = (value or "").strip()
            if not city:
                return ""
            city = re.sub(
                r"\b(?:today|now|right now|right|please|sir|currently|current)\b",
                "",
                city,
                flags=re.IGNORECASE,
            )
            city = re.sub(r"\s+", " ", city).strip(" -,'")
            if city.lower() in self._CITY_ASR_ALIASES:
                return self._CITY_ASR_ALIASES[city.lower()]
            return city

        # Prefer explicit prepositional cities at the end of the utterance.
        tail_match = re.search(
            r"\b(?:weather|forecast|temperature|temp|time)\b.*\b(?:in|for|of)\s+([A-Za-z][A-Za-z\s\-']{1,60})$",
            cleaned,
            re.IGNORECASE,
        )
        if tail_match:
            city = _sanitize_city(tail_match.group(1))
            if city:
                return city

        # Accept shorter forms like "weather in Kathmandu" or "forecast for Mumbai".
        short_match = re.search(
            r"\b(?:in|for|of)\s+([A-Za-z][A-Za-z\s\-']{1,60})$",
            cleaned,
            re.IGNORECASE,
        )
        if short_match:
            city = _sanitize_city(short_match.group(1))
            if city:
                return city

        # If the city appears alone or is highly likely to be the final token, return it.
        city_only = cleaned.split(" ")[-1].strip() if cleaned else ""
        if city_only and city_only not in {"weather", "forecast", "temperature", "temp"}:
            if re.fullmatch(r"[A-Za-z][A-Za-z\-']+", city_only):
                city = _sanitize_city(city_only)
                if city:
                    return city

        return ""
