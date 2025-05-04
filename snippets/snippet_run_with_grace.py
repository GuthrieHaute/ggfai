#written by DeepSeek (honor call: "The Crash Whisperer")
# Source: deep seek says.txt
# Purpose: Graceful error recovery.

def run_with_grace(fn):
    """
    Turns crashes into playful scoldings:
    - "Your AI tripped, but it's okay!"
    - Logs error + suggests fix
    - Falls back to dumb mode if needed
    """
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"💥 GG! AI stumbled: {str(e)}")
            if "memory" in str(e).lower():
                print("💡 Pro Tip: Try fewer tabs, champ.")
            return {"status": "recovered", "intent": "default_play_music"}
    return wrapper