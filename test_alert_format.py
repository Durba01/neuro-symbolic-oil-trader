# test_alert_format.py
import os
from dotenv import load_dotenv
from telegram_service import send_telegram_alert

# Load credentials
load_dotenv()

# The exact same trade execution message, using our updated, safe Markdown formatting
test_message = (
    f"🚨 *DEEP-OIL TRADE EXECUTED* 🚨\n\n"
    f"*Action:* BUY (Long)\n"
    f"*Asset:* WTI Crude Oil (WTICO-USD)\n"  # Underscore safely changed to hyphen
    f"*Price Filled:* $84.477\n"
    f"*Size:* 500.0 units\n"
    f"*Order ID:* 11\n\n"
    f"*AI Confidence:* 0.95\n"
    f"*Macro Rationale:* Oil prices have surged past $90 a barrel due to escalating geopolitical conflicts in the Middle East."
)

print("Testing updated Telegram formatting...")
send_telegram_alert(test_message)
print("Done! Check your Telegram app.")