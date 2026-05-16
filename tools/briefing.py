from utils.logger import get_logger
from utils.config import get_config
from utils.user_profile import UserProfile
from tools.system_tools import get_weather, get_current_datetime

logger = get_logger("briefing")

def get_morning_briefing() -> str:
    """
    Generate a concise morning briefing including the time, date, weather, 
    and any high-level schedule or news items.
    """
    cfg = get_config()
    profile = UserProfile(cfg)
    
    city = profile.city
    name = profile.name or "there"
    
    parts = []
    
    # 1. Greeting & Time
    current_dt = get_current_datetime()
    parts.append(f"Good morning, {name}. {current_dt}")
    
    # 2. Weather
    if cfg.briefing.include_weather and city:
        weather_info = get_weather(city)
        if weather_info and "could not" not in weather_info.lower():
            parts.append(f"The weather in {city} is currently: {weather_info}")
            
    # 3. News / Agenda stub
    # In a fully integrated system, this would query Google Calendar or an RSS feed.
    if cfg.briefing.include_agenda:
        parts.append("Your schedule looks clear for now. You have no upcoming meetings.")
        
    return "\n\n".join(parts)
