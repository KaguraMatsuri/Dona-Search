import httpx, trafilatura
from datetime import datetime
from ..interfaces import Document
from ..registry import register

@register("fetcher", "httpx")
class HTTPXFetcher:
    def __init__(self, timeout: float = 3.0, respect_robots: bool = True):
        self.timeout = timeout
        self.respect_robots = respect_robots
        self.client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": "DonaSearchBot/0.1 (+https://github.com/yourorg/dona-search)"},
            follow_redirects=True
        )

    def fetch(self, url: str) -> Document | None:
        try:
            resp = self.client.get(url)
            resp.raise_for_status()

            text = trafilatura.extract(resp.text, include_comments=False, include_tables=True)
            if not text or len(text) < 50:
                return None

            # Get title from HTML or use first line
            title = trafilatura.extract(resp.text, output_format="json")
            if title:
                title = eval(title).get("title", url.split("/")[-1])
            else:
                title = text.split("\n")[0][:100]

            return Document(
                url=url,
                title=title,
                text=text,
                fetched_at=datetime.utcnow().isoformat()
            )
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return None
