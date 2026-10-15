import os, httpx, json
from ..interfaces import SearchHit
from ..registry import register

@register("search", "brave")
class BraveSearchProvider:
    name = "brave"

    def __init__(self, api_key_env: str, top_k: int = 50):
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"{api_key_env} not set")
        self.top_k = top_k

    def search(self, query: str, k: int = 50) -> list[SearchHit]:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"X-Subscription-Token": self.api_key}
        params = {"q": query, "count": min(k, 20)}

        try:
            resp = httpx.get(url, headers=headers, params=params, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for i, item in enumerate(data.get("web", {}).get("results", []), 1):
                results.append(SearchHit(
                    rank=i,
                    title=item.get("title", ""),
                    url=item["url"],
                    snippet=item.get("description", ""),
                    provider="brave"
                ))
            return results
        except Exception as e:
            print(f"Brave search failed: {e}")
            return []
