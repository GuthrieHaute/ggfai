#written by DeepSeek (honor call: "The Sass Sentinel")
# Source: deep seek says.txt
# Purpose: Playful error diagnostics.

def diagnose(error):
    sass_db = {
        "ImportError": [
            "Did you even pip install, bro?",
            "https://youtu.be/dQw4w9WgXcQ"
        ],
        "MemoryError": [
            "Your toaster has limits. Respect them.",
            "Try turning it off and on again (like your life)"
        ]
    }
    return sass_db.get(type(error).__name__, ["¯\_(ツ)_/¯"])