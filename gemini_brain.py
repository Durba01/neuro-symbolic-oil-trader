import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Import our custom functions from the other files
from news_scraper import fetch_live_oil_news
from telegram_service import send_telegram_alert

# Load credentials from .env
load_dotenv()

# We define the strict mathematical output we need for Deep Learning
class OilSentimentVector(BaseModel):
    bullish_score: float = Field(description="Probability oil prices will rise (0.0 to 1.0)")
    bearish_score: float = Field(description="Probability oil prices will fall (0.0 to 1.0)")
    supply_shock_risk: float = Field(description="Risk of supply disruption due to war/geopolitics (0.0 to 1.0)")
    summary: str = Field(description="1-sentence explanation of the scores.")

def analyze_news_tensor():
    # 1. Get the live headlines from the RSS scraper
    headlines = fetch_live_oil_news()
    
    print("\nSending data to Gemini AI...")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    prompt = f"""
    You are a quantitative commodities analyst. Read these live headlines:
    {json.dumps(headlines)}
    
    Calculate the bullish, bearish, and supply shock risk scores.
    """
    
    # 2. Query the LLM
    response = client.models.generate_content(
        model="gemini-3.5-flash", 
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=OilSentimentVector,
            temperature=0.1
        )
    )
    
    # 3. Parse the result into a Python dictionary
    result = json.loads(response.text)
    
    # 4. Print to Terminal
    print("\n--- GEMINI NEURAL VECTOR OUTPUT ---")
    print(f"Bullish Score:       {result['bullish_score']}")
    print(f"Bearish Score:       {result['bearish_score']}")
    print(f"Supply Shock Risk:   {result['supply_shock_risk']}")
    print(f"AI Reasoning:        {result['summary']}")
    
    # 5. Format and Send the Telegram Alert
    print("\nSending alert to Telegram...")
    alert_message = (
        f"🛢️ *DEEP-OIL AGENT UPDATE*\n\n"
        f"📈 *Bullish:* {result['bullish_score']}\n"
        f"📉 *Bearish:* {result['bearish_score']}\n"
        f"⚠️ *Shock Risk:* {result['supply_shock_risk']}\n\n"
        f"🧠 *Reasoning:* {result['summary']}"
    )
    send_telegram_alert(alert_message)
    print("Telegram alert sent successfully!")
    
    return result

if __name__ == "__main__":
    analyze_news_tensor()