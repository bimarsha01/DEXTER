import subprocess
import urllib.parse
import webbrowser

from utils.logger import get_logger

logger = get_logger("media_tool")


def play_music(query: str, platform: str = "spotify") -> str:
    """Play music using platform deep links. No API key needed."""
    if not query:
        return "You must provide something to play."

    p = (platform or "spotify").lower().strip()
    encoded = urllib.parse.quote(query)
    web_encoded = urllib.parse.quote_plus(query)

    if p == "spotify":
        try:
            subprocess.Popen(
                ["cmd", "/c", "start", "", f"spotify:search:{encoded}"],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("music_playback_started", platform="spotify", query=query)
            return f"Searching Spotify for {query}."
        except Exception as e:
            logger.warning("music_playback_spotify_failed", error=str(e), exc_info=True)
            webbrowser.open(f"https://open.spotify.com/search/{encoded}")
            return f"Opening Spotify web for {query}."

    if p in {"youtube music", "youtube"}:
        webbrowser.open(f"https://music.youtube.com/search?q={web_encoded}")
        logger.info("music_playback_started", platform=p, query=query)
        return f"Searching YouTube Music for {query}."

    webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote_plus(f'{platform} {query}'.strip())}")
    logger.info("music_playback_started", platform=p or platform, query=query)
    return f"Searching {platform} for {query}."