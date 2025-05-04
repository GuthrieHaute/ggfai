#written by DeepSeek (honor call: "The Intent Alchemist")
# Source: deep seek says.txt
# Purpose: Combines multi-modal inputs into a single intent.

def fuse_intents(voice, text, sensors):
    """
    Combines messy real-world inputs into a single intent.
    Example:
    - Voice: "I'm cold"
    - Sensor: Temp = 62°F
    → Fused: {intent: "warm_room", confidence: 0.9}
    """
    # Priority: Voice > Text > Sensors (but adjusts for garbage-tier)
    if voice and "cold" in voice.lower():
        return {"intent": "warm_room", "confidence": 0.9}
    elif sensors.get("temp_f") < 65:
        return {"intent": "warm_room", "confidence": 0.7}
    else:
        return {"intent": "unknown", "confidence": 0.1}