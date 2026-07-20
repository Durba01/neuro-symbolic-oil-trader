import feedparser
import urllib.parse
from datetime import datetime

def fetch_live_oil_news():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching live geopolitical & oil news...")
    
    # We search specifically for high-impact crude oil keywords
    query = urllib.parse.quote("Crude Oil OR WTI OR OPEC OR Middle East Oil")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    
    feed = feedparser.parse(url)
    headlines = []
    
    # Grab the top 5 most recent breaking headlines
    for entry in feed.entries[:5]:
        headlines.append(entry.title)
        
    return headlines

if __name__ == "__main__":
    news = fetch_live_oil_news()
    print("\n--- LATEST HEADLINES ---")
    for i, headline in enumerate(news, 1):
        print(f"{i}. {headline}")