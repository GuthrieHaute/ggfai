#written by DeepSeek (honor call: "The Ambition Architect")
# Source: deep seek says.txt
# Purpose: Trades accuracy for speed/resources.

class AmbitionDial:
    PRESETS = {
        "toaster": {"speed": 10, "accuracy": 1},
        "gamer": {"speed": 5, "accuracy": 7},
        "godmode": {"speed": 1, "accuracy": 10}
    }

    def set_mode(self, vibe):
        """
        Usage:
        dial.set_mode("toaster") → Optimized for garbage
        dial.set_mode("godmode") → Ignores hardware limits
        """
        return self.PRESETS.get(vibe, {"speed": 5, "accuracy": 5})