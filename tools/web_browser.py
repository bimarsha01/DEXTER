import os
import re
import shutil
import subprocess
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

try:
    import winreg
except ImportError:  # pragma: no cover - only available on Windows
    winreg = None

logger = get_logger("web_browser")


_BROWSER_EXECUTABLES = {
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "edge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "firefox": "firefox.exe",
    "brave": "brave.exe",
}

_CONTENT_PLATFORM_URLS = {
    "youtube": "https://www.youtube.com/results?search_query={query}",
    "youtube music": "https://music.youtube.com/search?q={query}",
    "spotify": "https://open.spotify.com/search/{query}",
    "soundcloud": "https://soundcloud.com/search/sounds?q={query}",
    "apple music": "https://music.apple.com/us/search?term={query}",
    "netflix": "https://www.netflix.com/search?q={query}",
    "prime video": "https://www.primevideo.com/search/ref=atv_nb_sr?phrase={query}",
    "espn": "https://www.espn.com/search/results?q={query}",
    "twitch": "https://www.twitch.tv/search?term={query}",
}


def _resolve_browser_executable(browser: str) -> str:
    browser_key = browser.lower().strip()
    exe_name = _BROWSER_EXECUTABLES.get(browser_key)
    if not exe_name:
        return ""

    which_path = shutil.which(exe_name)
    if which_path:
        return which_path

    if winreg is not None:
        reg_path = rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{exe_name}"
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(hive, reg_path) as key:
                    value, _ = winreg.QueryValueEx(key, "")
                    if value and os.path.exists(value):
                        return value
            except OSError:
                pass

    candidate_roots = [
        os.environ.get("LOCALAPPDATA", ""),
        os.environ.get("PROGRAMFILES", ""),
        os.environ.get("PROGRAMFILES(X86)", ""),
    ]
    candidate_paths = {
        "chrome.exe": [("Google", "Chrome", "Application", "chrome.exe")],
        "msedge.exe": [("Microsoft", "Edge", "Application", "msedge.exe")],
        "firefox.exe": [("Mozilla Firefox", "firefox.exe")],
        "brave.exe": [("BraveSoftware", "Brave-Browser", "Application", "brave.exe")],
    }

    for root in candidate_roots:
        if not root:
            continue
        for parts in candidate_paths.get(exe_name, []):
            candidate = Path(root, *parts)
            if candidate.exists():
                return str(candidate)

    return ""


def _normalize_platform_name(platform: str) -> str:
    cleaned = (platform or "").strip().lower()
    cleaned = cleaned.replace("https://", "").replace("http://", "")
    cleaned = cleaned.split("/")[0]
    cleaned = cleaned.replace("www.", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    aliases = {
        "youtube music": "youtube music",
        "yt music": "youtube music",
        "youtube": "youtube",
        "spotify": "spotify",
        "soundcloud": "soundcloud",
        "apple music": "apple music",
        "music.apple": "apple music",
        "netflix": "netflix",
        "prime video": "prime video",
        "amazon prime video": "prime video",
        "espn": "espn",
        "twitch": "twitch",
    }
    return aliases.get(cleaned, cleaned)


def _guess_default_platform(query: str) -> str:
    text = (query or "").lower()
    if any(keyword in text for keyword in ["podcast", "episode", "interview"]):
        return "spotify"
    if any(keyword in text for keyword in ["movie", "film", "series", "show", "tv"]):
        return "netflix"
    if any(keyword in text for keyword in ["highlight", "highlights", "sport", "sports", "match", "f1", "nfl", "nba", "ucl"]):
        return "youtube"
    if any(keyword in text for keyword in ["song", "music", "playlist", "album", "track", "artist", "lo-fi", "lofi"]):
        return "youtube music"
    return "youtube"


def _build_content_search_url(query: str, platform: str = "", content_type: str = "") -> str:
    platform_key = _normalize_platform_name(platform)
    if not platform_key:
        platform_key = _guess_default_platform(query if query else content_type)

    encoded_query = urllib.parse.quote_plus(query)
    template = _CONTENT_PLATFORM_URLS.get(platform_key)
    if template:
        return template.format(query=encoded_query)

    if platform_key.startswith("http://") or platform_key.startswith("https://"):
        return f"{platform_key.rstrip('/')}/search?q={encoded_query}"

    if "." in platform_key:
        return f"https://{platform_key.rstrip('/')}/search?q={encoded_query}"

    # Reasonable fallback for unknown platforms: search the web for the platform + title.
    fallback_query = urllib.parse.quote_plus(f"{platform_key} {query}".strip())
    return f"https://www.google.com/search?q={fallback_query}"

def search_google(query: str) -> str:
    """
    Opens a new tab in the user's default browser and performs a Google search for the query.
    """
    if not query:
        return "You must provide a search term."
        
    logger.info("browser_google_search", query_length=len(query))
    
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://www.google.com/search?q={encoded_query}"
    
    try:
         webbrowser.open(url)
         return f"Successfully searched Google for '{query}'"
    except Exception as e:
         logger.error("browser_open_failed", error=str(e), exc_info=True)
         return f"Error executing search: {e}"

def open_url(url: str) -> str:
    """
    Properly formats a URL string (adding https:// if missed) and opens it.
    """
    if not url.startswith("http"):
         url = f"https://{url}"
         
    parsed = urllib.parse.urlparse(url)
    logger.info("browser_open_url", scheme=parsed.scheme or "", netloc=parsed.netloc or "")
    
    try:
        webbrowser.open(url)
        return f"Successfully opened {url}"
         
    except Exception as e:
         logger.error("browser_open_url_failed", error=str(e), exc_info=True)
         return f"Failed to open the requested URL."


def open_url_in_browser(url: str, browser: str) -> str:
    """
    Opens a URL in a specific browser (chrome, edge, firefox, brave).
    """
    if not url:
        return "You must provide a URL to open."
    if not browser:
        return open_url(url)

    if not url.startswith("http"):
        url = f"https://{url}"

    browser_key = browser.lower().strip()
    exe = _resolve_browser_executable(browser_key)
    if not exe:
        fallback = open_url(url)
        return f"I could not find {browser_key}. {fallback}"

    try:
        subprocess.Popen([exe, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Successfully opened {url} in {browser_key}."
    except Exception as e:
        logger.error("browser_open_failed", error=str(e), exc_info=True)
        return f"Failed to open {url} in {browser_key}."

def search_youtube(search_term: str) -> str:
    """
    Searches YouTube in the browser for the given term, automatically navigating to the top results hook.
    """
    return search_content_platform(search_term, platform="youtube", content_type="video")


def search_content_platform(query: str, platform: str = "", content_type: str = "", browser: Optional[str] = "") -> str:
    """
    Search for media/content on a specific platform or a sensible default platform.

    If the platform is unknown, the function falls back to a general web search that
    combines the platform name and query so Dexter can still do something reasonable.
    """
    if not query:
        return "You must provide something to search for."

    platform_key = _normalize_platform_name(platform)
    logger.info(
        "browser_content_search",
        platform=platform_key or "",
        content_type=(content_type or "").strip().lower(),
        query_length=len(query),
    )

    url = _build_content_search_url(query, platform=platform_key, content_type=content_type)

    try:
        if browser:
            return open_url_in_browser(url, browser)
        webbrowser.open(url)
        if platform_key:
            return f"Successfully opened {platform_key} search for '{query}'."
        return f"Successfully opened a search for '{query}'."
    except Exception as e:
        logger.error("browser_content_search_failed", error=str(e), exc_info=True)
        return "Failed to open the requested content search."