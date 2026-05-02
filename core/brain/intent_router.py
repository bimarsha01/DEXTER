import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from utils.config import DexterConfig


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
    kind: str  # confirm | slot
    tool_name: str
    args: Dict
    prompt: str
    expires_at: float


class IntentRouter:
    def __init__(self, config: DexterConfig):
        self.default_city = (config.defaults.city or "").strip()

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

        return IntentDecision(action="ask", prompt=pending.prompt)

    def detect_intent(self, text: str) -> IntentDecision:
        lowered = text.lower().strip()

        # Vision intents
        if any(kw in lowered for kw in ["look at", "see", "screen", "screenshot", "what's on my screen"]):
            return IntentDecision(action="vision", vision_mode="screen")

        file_match = re.search(r"([\w\-./\\]+\.(py|txt|md|json|yaml|yml))", text, re.IGNORECASE)
        if file_match:
            return IntentDecision(action="vision", vision_mode="file", file_path=file_match.group(1))
        if any(kw in lowered for kw in ["function", "code", "bug", "error in this file"]):
            return IntentDecision(action="ask", vision_mode="file", prompt="Which file should I inspect, sir? Provide a relative path.")

        # Weather
        if "weather" in lowered or "forecast" in lowered:
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
        app_match = re.match(r"(?:open|launch|start)\s+(.+)$", lowered)
        if app_match:
            app_name = app_match.group(1).strip()
            if app_name:
                return IntentDecision(action="tool", tool_name="open_application", args={"app_name": app_name})
            return IntentDecision(action="ask", tool_name="open_application", prompt="Which application should I open, sir?")

        # Clipboard
        if "clipboard" in lowered and any(kw in lowered for kw in ["read", "show", "what's on", "what is on", "check"]):
            return IntentDecision(action="tool", tool_name="read_clipboard", args={})

        copy_match = re.match(r"(?:copy|set clipboard)\s+(.+)$", text, re.IGNORECASE)
        if copy_match:
            content = copy_match.group(1).strip()
            if content:
                return IntentDecision(action="tool", tool_name="copy_to_clipboard", args={"text": content})
            return IntentDecision(action="ask", tool_name="copy_to_clipboard", prompt="What should I copy to the clipboard, sir?")

        return IntentDecision(action="none")

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
        match = re.search(r"in\s+([A-Za-z\s]+)$", text)
        if match:
            return match.group(1).strip()
        return ""
