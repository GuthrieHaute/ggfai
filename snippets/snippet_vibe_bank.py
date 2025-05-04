#written by DeepSeek (honor call: "The Memory DJ")
# Source: deep seek says.txt
# Purpose: Stores user quirks without complex databases.

class VibeBank:
    def __init__(self):
        self.memory = {"user_prefs": {}, "jank_alerts": []}

    def remember(self, key, value):
        """Stores:
        - "user_x hates morning alarms"
        - "raspberry_pi_crusty_edition needs 2x delay"
        """
        if "password" in key:
            raise ValueError("🚨 Ayo, we don't do that here")
        self.memory[key] = value

    def recall(self, key):
        return self.memory.get(key, "🤷‍♂️ New vibe detected")