"""
Platform-agnostic media playback without external API dependencies.
Leverages OS-native URI scheme detection and deep links to launch
music applications (Spotify, Apple Music) if installed, and falls back
to browser-based web players otherwise.
"""

import subprocess
import urllib.parse
import webbrowser

from utils.logger import get_logger

logger = get_logger("media_tool")


def is_uri_handler_registered(protocol: str) -> bool:
    """Check if a URL protocol handler (e.g., 'spotify', 'music') is registered in the Windows Registry."""
    try:
        import winreg
        # We only need to check if the key exists in HKEY_CLASSES_ROOT
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, protocol):
            return True
    except Exception:
        return False


def play_music(query: str, platform: str = "spotify") -> str:
    """Play music using native desktop apps (if installed) or web player fallbacks. No API key needed."""
    if not query:
        return "You must provide something to play."

    p = (platform or "spotify").lower().strip()
    encoded = urllib.parse.quote(query)
    web_encoded = urllib.parse.quote_plus(query)

    if p == "spotify":
        # Hybrid Approach: Try Production OAuth API first
        try:
            from tools.spotify_tool import play_spotify_music
            api_result = play_spotify_music(query)
            if "not configured" not in api_result:
                logger.info("music_playback_started_api", platform="spotify", query=query)
                return api_result
        except Exception as e:
            logger.debug("music_playback_api_skipped", error=str(e))

        # Fallback to OS Deep-link
        if is_uri_handler_registered("spotify"):
            try:
                subprocess.Popen(
                    ["cmd", "/c", "start", "", f"spotify:search:{encoded}"],
                    shell=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("music_playback_started_deeplink", platform="spotify", query=query)
                return f"Opening Spotify desktop app to search for {query}."
            except Exception as e:
                logger.warning("music_playback_spotify_deeplink_failed", error=str(e), exc_info=True)
                
        # Fallback to web
        webbrowser.open(f"https://open.spotify.com/search/{encoded}")
        return f"Opening Spotify web player for {query}."

    if p in {"apple music", "apple"}:
        if is_uri_handler_registered("music"):
            try:
                subprocess.Popen(
                    ["cmd", "/c", "start", "", f"music://search?term={encoded}"],
                    shell=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logger.info("music_playback_started_deeplink", platform="apple music", query=query)
                return f"Opening Apple Music desktop app for {query}."
            except Exception as e:
                logger.warning("music_playback_apple_deeplink_failed", error=str(e), exc_info=True)
                
        # Fallback to web
        webbrowser.open(f"https://music.apple.com/search?term={web_encoded}")
        return f"Opening Apple Music web player for {query}."

    if p in {"youtube music", "youtube", "ytmusic"}:
        webbrowser.open(f"https://music.youtube.com/search?q={web_encoded}")
        logger.info("music_playback_started", platform=p, query=query)
        return f"Opening YouTube Music for {query}."

    # Generic search fallback
    webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote_plus(f'{platform} {query}'.strip())}")
    logger.info("music_playback_started", platform=p or platform, query=query)
    return f"Searching {platform} for {query}."


def pause_music() -> str:
    """Pause or resume the currently playing system media using a simulated OS media key."""
    try:
        import pyautogui
        pyautogui.press('playpause')
        return "Toggled play/pause for system media."
    except ImportError:
        return "The 'pyautogui' module is missing. Please install it to use global media keys."
    except Exception as e:
        logger.error("media_key_pause_failed", error=str(e))
        return f"Could not toggle play/pause: {str(e)}"


def next_track() -> str:
    """Skip to the next track using a simulated OS media key."""
    try:
        import pyautogui
        pyautogui.press('nexttrack')
        return "Skipped to the next track."
    except ImportError:
        return "The 'pyautogui' module is missing. Please install it to use global media keys."
    except Exception as e:
        logger.error("media_key_next_failed", error=str(e))
        return f"Could not skip track: {str(e)}"
