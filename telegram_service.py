import os
import requests
from dotenv import load_dotenv

load_dotenv()

def send_telegram_alert(message: str):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("Telegram keys missing. Skipping alert.")
        return
        
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        
        # Check if Telegram rejected the message
        if response.status_code != 200:
            print(f"\n❌ TELEGRAM REJECTED THE MESSAGE!")
            print(f"Error Code: {response.status_code}")
            print(f"Details: {response.text}\n")
        else:
            print("✅ Telegram notification successfully delivered to your phone!")
            
    except Exception as e:
        print(f"Failed to connect to Telegram: {e}")