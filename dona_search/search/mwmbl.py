from urllib.parse import urlsplit

import httpx

from ..config import ProviderConfig
from ..interfaces import ProviderResult, SearchHit
from ..registry import register


@register('search', 'mwmbl')
class MwmblProvider:
    def __init__(self, config: ProviderConfig, client: httpx.Client):
        self.config, self.client, self.name = config, client, config.name

    def search(self, query: str, k: int) -> ProviderResult:
        response = self.client.get(
            self.config.search_url or 'https://api.mwmbl.org/api/v1/search/',
            params={**self.config.params, 's': query}, follow_redirects=False,
        )
        response.raise_for_status()
        results = parse_results(response.json(), self.name)[:k]
        return ProviderResult(self.name, 'ok' if results else 'empty', results, response.status_code)


def parse_results(data, provider: str = 'mwmbl') -> list[SearchHit]:
    if not isinstance(data, list):
        raise ValueError('invalid_mwmbl_response')
    hits = []
    for rank, item in enumerate(data, 1):
        url = item['url']
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            continue
        hits.append(SearchHit(
            rank, ''.join(p['value'] for p in item['title']), url,
            ''.join(p['value'] for p in item.get('extract', [])), provider,
        ))
    return hits
