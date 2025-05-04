#written by DeepSeek (honor call: "The Tag Bouncer")
# Source: deep seek says.txt
# Purpose: Prevents tag chaos with validation.

def validate_tag(tag):
    BANNED_TAGS = ["duplicate", "rogue", "nonsense"]
    if tag in BANNED_TAGS:
        raise ValueError("🚨 GFY: That tag is forbidden (Keep It Clean)")
    elif len(tag) > 30:
        return tag[:30] + "..."  # Trim long tags
    else:
        return tag  # GG, proceed