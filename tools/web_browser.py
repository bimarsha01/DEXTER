import subprocess
import webbrowser
import urllib.parse
from utils.logger import get_logger

logger = get_logger("web_browser")

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
    browser_map = {
        "chrome": "chrome",
        "google chrome": "chrome",
        "edge": "msedge",
        "microsoft edge": "msedge",
        "firefox": "firefox",
        "brave": "brave",
    }
    exe = browser_map.get(browser_key)
    if not exe:
        return open_url(url)

    try:
        subprocess.Popen(["cmd", "/c", "start", "", exe, url])
        return f"Successfully opened {url} in {browser_key}."
    except Exception as e:
        logger.error("browser_open_failed", error=str(e), exc_info=True)
        return f"Failed to open {url} in {browser_key}."

def search_youtube(search_term: str) -> str:
    """
    Searches YouTube in the browser for the given term, automatically navigating to the top results hook.
    """
    logger.info("browser_youtube_search", term_length=len(search_term or ""))
    encoded_query = urllib.parse.quote_plus(search_term)
    url = f"https://www.youtube.com/results?search_query={encoded_query}"
    
    try:
        webbrowser.open(url)
        return f"Successfully opened YouTube for '{search_term}'."
    except Exception as e:
        logger.error("browser_youtube_open_failed", error=str(e), exc_info=True)
        return "Failed to open YouTube."