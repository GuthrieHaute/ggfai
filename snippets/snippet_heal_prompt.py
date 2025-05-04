#written by DeepSeek (honor call: "The Prompt Paramedic")
# Source: deep seek says.txt
# Purpose: Fixes bad inputs before processing.

def heal_prompt(user_input):
    """
    Fixes:
    - "play musac" → "play music"
    - "im cold" → "I'm cold"
    - "BRRRR" → "warm_room"
    """
    fixes = {
        "musac": "music",
        "im cold": "I'm cold",
        "brrr": "warm_room"
    }
    return fixes.get(user_input.lower(), user_input)