# eia_parser.py
import os
import json
import feedparser
import urllib.parse
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

class EIANewsVector(BaseModel):
    inventory_change_millions: float = Field(description="The actual change in US crude oil inventories in millions of barrels (e.g., -4.2 for a 4.2M barrel draw, +1.5 for a 1.5M build). Use 0.0 if not found.")
    is_new_release: bool = Field(description="True if the headlines represent a newly released weekly report (usually within the last 24-48 hours).")
    fundamental_impact: str = Field(description="Must be 'BULLISH' (larger draw than expected or unexpected draw), 'BEARISH' (larger build than expected or unexpected build), or 'NEUTRAL'.")
    summary: str = Field(description="A brief 1-sentence summary of the weekly inventory report.")

def fetch_eia_news():
    """Specifically queries Google News RSS for the weekly EIA inventory reports."""
    query = urllib.parse.quote('eia weekly crude oil inventories report')
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    feed = feedparser.parse(url)
    headlines = [entry.title for entry in feed.entries[:5]]
    return headlines

def analyze_eia_report():
    print("📊 Fetching and analyzing latest weekly EIA Petroleum Status Report...")
    headlines = fetch_eia_news()
    
    if not headlines:
        print("No EIA headlines found.")
        return None

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    prompt = f"""
    You are a commodities macro analyst. Read these raw headlines regarding the latest weekly US EIA Crude Oil Inventory report:
    {json.dumps(headlines)}
    
    Extract the actual inventory change (in millions of barrels) and determine if it represents a newly released report (within the last 24-48 hours) and its fundamental market impact.
    """
    
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EIANewsVector,
            temperature=0.0
        )
    )
    
    result = json.loads(response.text)
    
    # Save results locally to eia_fundamental.json
    with open("eia_fundamental.json", "w") as f:
        json.dump(result, f, indent=2)
        
    print("\n--- EIA WEEKLY FUNDAMENTAL VECTOR ---")
    print(f"Inventory Change:  {result['inventory_change_millions']}M barrels")
    print(f"New Weekly Report: {result['is_new_release']}")
    print(f"Market Impact:     {result['fundamental_impact']}")
    print(f"AI Summary:        {result['summary']}")
    
    return result

if __name__ == "__main__":
    analyze_eia_report()