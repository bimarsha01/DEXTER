import webbrowser
import urllib.parse
from utils.logger import logger

def search_google(query: str) -> str:
    """
    Opens a new tab in the user's default browser and performs a Google search for the query.
    """
    if not query:
        return "You must provide a search term."
        
    logger.info(f"Googling: {query}")
    
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://www.google.com/search?q={encoded_query}"
    
    try:
         webbrowser.open(url)
         return f"Successfully searched Google for '{query}'"
    except Exception as e:
         logger.error(f"Failed to open browser: {e}")
         return f"Error executing search: {e}"

def open_url(url: str) -> str:
    """
    Properly formats a URL string (adding https:// if missed) and opens it.
    """
    if not url.startswith("http"):
         url = f"https://{url}"
         
    logger.info(f"Navigating to {url}")
    
    try:
        webbrowser.open(url)
        return f"Successfully opened {url}"
         
    except Exception as e:
         return f"Failed to open the requested URL."

def search_youtube(search_term: str) -> str:
    """
    Searches YouTube in the browser for the given term, automatically navigating to the top results hook.
    """
    logger.info(f"Searching YouTube for: {search_term}")
    encoded_query = urllib.parse.quote_plus(search_term)
    url = f"https://www.youtube.com/results?search_query={encoded_query}"
    
    try:
        webbrowser.open(url)
        return f"Successfully opened YouTube for '{search_term}'."
    except Exception as e:
        return "Failed to open YouTube."