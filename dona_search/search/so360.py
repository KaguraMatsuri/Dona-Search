from urllib.parse import urlsplit

import httpx
from lxml import html

from ..config import ProviderConfig
from ..interfaces import ProviderResult, SearchHit
from ..registry import register


@register('search', 'so360')
class So360Provider:
    def __init__(self, config: ProviderConfig, client: httpx.Client):
        self.config, self.client, self.name = config, client, config.name

    def search(self, query: str, k: int) -> ProviderResult:
        response = self.client.get(
            self.config.search_url or 'https://www.so.com/s',
            params={'pn': '1', **self.config.params, 'q': query},
            follow_redirects=False,
        )
        response.raise_for_status()
        results = parse_results(response.text, self.name)[:k]
        return ProviderResult(self.name, 'ok' if results else 'empty', results, response.status_code)


def parse_results(text: str, provider: str = '360') -> list[SearchHit]:
    tree = html.fromstring(text)
    hits = []
    for rank, item in enumerate(tree.xpath('//li[contains(concat(" ", normalize-space(@class), " "), " res-list ")]'), 1):
        anchors = item.xpath('.//h3[contains(@class, "res-title")]/a')
        if not anchors:
            continue
        anchor = anchors[0]
        url = anchor.get('data-mdurl') or anchor.get('href') or ''
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname:
            continue
        if parsed.hostname == 'ai.so.com' or parsed.hostname.endswith('.ai.so.com'):
            continue
        paragraphs = item.xpath('.//p[@class="res-desc"]|.//span[@class="res-list-summary"]')
        hits.append(SearchHit(
            rank, ' '.join(anchor.text_content().split()), url,
            ' '.join(' '.join(p.text_content() for p in paragraphs).split()), provider,
        ))
    return hits
