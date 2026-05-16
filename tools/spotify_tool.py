import os
import urllib.parse
from typing import Optional, Any

from utils.config import get_config, get_workspace_root
from utils.logger import get_logger

logger = get_logger("spotify_tool")


def _get_spotify_client() -> Optional[Any]:
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth, CacheFileHandler
    except ImportError:
        logger.warning("spotipy_not_installed")
        return None

    config = get_config()
    client_id = config.spotify.client_id
    client_secret = config.spotify.client_secret

    if not client_id or not client_secret:
        return None

    try:
        data_dir = os.path.join(get_workspace_root(), "data")
        os.makedirs(data_dir, exist_ok=True)
        cache_path = os.path.join(data_dir, ".spotify_cache.json")
        
        # We explicitly use open_browser=True so the user gets a one-click login pop-up
        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=config.spotify.redirect_uri,
            scope="user-read-playback-state,user-modify-playback-state",
            open_browser=True,
            cache_handler=CacheFileHandler(cache_path=cache_path)
        )
        return spotipy.Spotify(auth_manager=auth_manager)
    except Exception as e:
        logger.error("spotify_auth_failed", error=str(e), exc_info=True)
        return None


def play_spotify_music(query: str) -> str:
    """Search and instantly play music using the Spotify Web API."""
    sp = _get_spotify_client()
    if not sp:
        return "Spotify API is not configured. Falling back to OS routing."

    try:
        devices = sp.devices()
        active_device = next((d for d in devices.get("devices", []) if d.get("is_active")), None)
        if not active_device and devices.get("devices"):
            active_device = devices["devices"][0]

        device_id = active_device["id"] if active_device else None

        results = sp.search(q=query, limit=1, type="track,artist,album,playlist")
        
        uri = None
        name = ""
        type_str = ""

        if results.get("tracks", {}).get("items"):
            item = results["tracks"]["items"][0]
            uri = item["uri"]
            name = f"{item['name']} by {item['artists'][0]['name']}"
            type_str = "track"
        elif results.get("artists", {}).get("items"):
            item = results["artists"]["items"][0]
            uri = item["uri"]
            name = item["name"]
            type_str = "artist"
        elif results.get("albums", {}).get("items"):
            item = results["albums"]["items"][0]
            uri = item["uri"]
            name = item["name"]
            type_str = "album"
        elif results.get("playlists", {}).get("items"):
            item = results["playlists"]["items"][0]
            uri = item["uri"]
            name = item["name"]
            type_str = "playlist"

        if not uri:
            return f"Could not find anything for '{query}' on Spotify."

        if type_str == "track":
            sp.start_playback(device_id=device_id, uris=[uri])
        else:
            sp.start_playback(device_id=device_id, context_uri=uri)

        device_name = active_device['name'] if active_device else 'your active device'
        return f"Playing {type_str} '{name}' on {device_name}."

    except Exception as e:
        msg = str(e)
        logger.error("spotify_playback_failed", error=msg, exc_info=True)
        if "NO_ACTIVE_DEVICE" in msg:
            return f"Could not play '{query}'. Please open the Spotify app on one of your devices first so I can control it."
        return f"Failed to play music via Spotify API: {msg}"
