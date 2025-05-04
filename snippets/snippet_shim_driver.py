#written by DeepSeek (honor call: "The Jank Bender")
# Source: deep seek says.txt
# Purpose: Adapts unsupported hardware.

def shim_driver(device):
    """
    Hacks unsupported hardware into compliance:
    - Wii Remote → Gesture controller
    - Old Kindle → Text input
    - USB fan → "Yes, we can pretend it's a mic"
    """
    if "nintendo" in device.lower():
        return {"type": "gesture", "confidence": 0.6}
    elif "e-ink" in device.lower():
        return {"type": "text", "refresh_rate": "slow_af"}
    else:
        return {"type": "unknown", "vibe": "experimental"}