import subprocess
import shlex
from typing import Optional

from utils.logger import get_logger

try:
    from tools import web_browser
except Exception as e:
    temp_logger = get_logger("youtube_tool")
    temp_logger.warning("youtube_web_browser_import_failed", error=str(e), exc_info=True)
    web_browser = None

logger = get_logger("youtube_tool")


def _is_url(text: str) -> bool:
    return text.startswith("http://") or text.startswith("https://")


def play_youtube(query: str, browser: Optional[str] = "chrome", mode: Optional[str] = "browser") -> str:
    """
    Resolve a YouTube query to the top video and open it in the specified browser.

    - `query`: search query or a direct YouTube URL/video id.
    - `browser`: browser name (chrome, edge, firefox, brave). If omitted opens default browser.
    - `mode`: currently only supports `browser` (open URL). Kept for future `stream` mode.

    Falls back to opening the YouTube search page if `yt-dlp` is unavailable or fails.
    """
    if not query:
        return "You must provide a search term."

    # If the user already provided a URL, open directly
    if _is_url(query) or query.startswith("www."):
        url = query if _is_url(query) else f"https://{query}"
        if web_browser:
            return web_browser.open_url_in_browser(url, browser)
        return f"Opening {url}"

    # Only browser mode implemented for now
    if mode != "browser":
        return "Only 'browser' mode is supported right now."

    # Try to resolve top result via yt-dlp CLI
    try:
        cmd = f"yt-dlp --get-id \"ytsearch1:{query}\""
        proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=15)
        if proc.returncode == 0 and proc.stdout:
            video_id = proc.stdout.strip().splitlines()[0]
            url = f"https://www.youtube.com/watch?v={video_id}"
            if web_browser:
                return web_browser.open_url_in_browser(url, browser)
            return f"Opening {url}"
        else:
            logger.warning("yt_dlp_failed", cmd=cmd, rc=proc.returncode, err=proc.stderr[:200])
    except FileNotFoundError:
        logger.info("yt_dlp_not_found")
    except subprocess.SubprocessError as exc:
        logger.warning("yt_dlp_error", error=str(exc))

    # Fallback: open YouTube search page
    if web_browser:
        return web_browser.search_youtube(query)
    return f"Please open YouTube and search for: {query}"
