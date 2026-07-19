"""
Quick standalone diagnostic - run this on your own machine (same conda
env, so feedparser is available) to see exactly which of the 5 RSS
feeds are failing right now, and why.

Usage:
    conda activate stock_predictor
    python diagnose_rss_feeds.py
"""
import urllib.request
import feedparser

FEEDS = {
    "Bernama":               "https://www.bernama.com/en/rssfeed.php",
    "Free Malaysia Today":   "https://www.freemalaysiatoday.com/feeds/rss/headlines/",
    "Malay Mail":            "https://www.malaymail.com/feed/rss/malaysia",
    "The Sun Malaysia":      "https://thesun.my/feed/",
    # --- candidates to replace the dead theedgemalaysia.com/rss.html ---
    "[candidate] The Edge Markets - Top Stories": "https://feeds.theedgemarkets.com/theedgemarkets/mytopstories.rss",
    "[candidate] The Star - RSS":                 "https://www.thestar.com.my/rss/editorspick",
    "[candidate] New Straits Times - Business":   "https://www.nst.com.my/rss/business",
}

TIMEOUT = 8

for name, url in FEEDS.items():
    print(f"\n{name}  ->  {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            status = resp.status
            raw = resp.read()
        parsed = feedparser.parse(raw)
        n_entries = len(parsed.entries)
        bozo = getattr(parsed, "bozo", 0)
        print(f"  HTTP {status} | {n_entries} entries | bozo={bozo}"
              + (f" | parse issue: {parsed.bozo_exception}" if bozo else ""))
    except urllib.error.HTTPError as e:
        print(f"  FAILED - HTTP error {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        print(f"  FAILED - URL/connection error: {e.reason}")
    except Exception as e:
        print(f"  FAILED - {type(e).__name__}: {e}")