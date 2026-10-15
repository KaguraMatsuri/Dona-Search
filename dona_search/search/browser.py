import base64
from copy import deepcopy
import shutil
import socket
import stat
import tempfile
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit

from lxml import etree, html

from ..config import ExtractionRules
from ..interfaces import ProviderResult, SearchHit
from ..registry import register


PRESETS = {
    'search_google': ('https://www.google.com/search', ExtractionRules(
        item_xpath="//div[contains(@class,'MjjYud')][.//a//h3][not(.//div[contains(@class,'MjjYud')][.//a//h3])]",
        title_xpath='.//h3[1]', url_xpath='.//a[.//h3][1]/@href',
        snippet_xpath=".//*[contains(@class,'VwiC3b')]|.//*[contains(@class,'IsZvec')][not(.//*[contains(@class,'VwiC3b')])]",
        query_xpath="//textarea[@name='q']|//input[@name='q']/@value",
        empty_xpath="//*[@id='topstuff'][contains(.,'did not match any documents') or contains(.,'找不到和') or contains(.,'没有找到')]",
        blocked_xpath="//form[contains(@action,'sorry')]|//iframe[contains(@src,'recaptcha')]",
    )),
    'search_bing': ('https://www.bing.com/search', ExtractionRules(
        item_xpath="//li[contains(concat(' ',normalize-space(@class),' '),' b_algo ')]",
        title_xpath='.//h2[1]', url_xpath='.//h2/a[1]/@href',
        snippet_xpath=".//div[contains(@class,'b_caption')]/p",
        query_xpath="//input[@name='q']/@value",
        empty_xpath="//*[@id='b_results']//*[contains(@class,'b_no')]",
        blocked_xpath="//*[@id='b_captcha']|//iframe[contains(@src,'captcha')]",
    )),
    'search_yandex': ('https://yandex.com/search/', ExtractionRules(
        item_xpath="//*[self::li or self::div][contains(concat(' ',normalize-space(@class),' '),' serp-item ')][.//h2]",
        title_xpath='.//h2[1]', url_xpath='.//h2//a[1]/@href',
        snippet_xpath=".//*[contains(@class,'OrganicText') or contains(@class,'text-container')][not(.//*[contains(@class,'OrganicText') or contains(@class,'text-container')])]",
        query_xpath="//input[@name='text']/@value",
        empty_xpath="//*[contains(@class,'EmptySearchResults')]",
        blocked_xpath="//form[contains(@action,'checkcaptcha')]|//*[contains(@class,'CheckboxCaptcha')]",
    )),
}


def _text(value):
    if isinstance(value, list):
        return ' '.join(_text(v) for v in value).strip()
    if isinstance(value, etree._Element):
        return ' '.join(value.itertext()).strip()
    return str(value).strip() if value is not None else ''


def target_url(value, base, engine):
    url = urljoin(base, value)
    parsed = urlsplit(url)
    params = parse_qs(parsed.query)
    host = parsed.hostname or ''
    if engine == 'search_google' and (host == 'google.com' or host.endswith('.google.com')) and parsed.path == '/url':
        url = (params.get('q') or params.get('url') or [''])[0]
    elif engine == 'search_bing' and (host == 'bing.com' or host.endswith('.bing.com')) and parsed.path == '/ck/a':
        encoded = params.get('u', [''])[0]
        if encoded.startswith('a1'):
            try:
                url = base64.urlsafe_b64decode(encoded[2:] + '=' * (-len(encoded[2:]) % 4)).decode()
            except (ValueError, UnicodeDecodeError):
                return None
        else:
            return None
    parsed = urlsplit(url)
    return url if parsed.scheme in ('http', 'https') and parsed.hostname and not parsed.username and not parsed.password else None


class BrowserSearchProvider:
    def __init__(self, config, browser):
        self.config, self.browser, self.name = config, browser, config.name
        preset = PRESETS.get(config.type)
        self.search_url = config.search_url or preset[0]
        self.rules = config.rules or preset[1]
        self.xpath = {key: etree.XPath(value) for key, value in self.rules.model_dump().items() if value}

    def search(self, query, k):
        parsed = urlsplit(self.search_url)
        params = dict(parse_qs(parsed.query))
        params.update(self.config.params)
        params[self.config.query_param] = query
        url = urlunsplit(parsed._replace(query=urlencode(params, doseq=True)))
        return self.browser.search(self, query, url, k)

    def inspect(self, source, page_url, query, k):
        location = urlsplit(page_url)
        if location.hostname == 'consent.google.com':
            return ProviderResult(self.name, 'blocked', error_code='consent_required')
        if location.path.startswith(('/sorry', '/showcaptcha', '/checkcaptcha')):
            return ProviderResult(self.name, 'blocked', error_code='captcha_required')
        tree = html.fromstring(source)
        if self.xpath.get('blocked_xpath') is not None and self.xpath['blocked_xpath'](tree):
            return ProviderResult(self.name, 'blocked', error_code='captcha_required')
        if self.xpath.get('query_xpath') is not None:
            values = self.xpath['query_xpath'](tree)
            echoed = next((_text(v) for v in values if _text(v)), '') if isinstance(values, list) else _text(values)
            if echoed and ' '.join(echoed.split()) != ' '.join(query.split()):
                return ProviderResult(self.name, 'query_mismatch', error_code='query_echo_mismatch')
        hits = []
        items = self.xpath['item_xpath'](tree)
        if not isinstance(items, list) or any(not isinstance(item, etree._Element) for item in items):
            raise ValueError('invalid_result_items')
        for rank, item in enumerate(items, 1):
            # A detached subtree is the document boundary for every field XPath.
            item = deepcopy(item)
            item.tail = None
            title = _text(self.xpath['title_xpath'](item))
            url_values = self.xpath['url_xpath'](item)
            if isinstance(url_values, list):
                url_values = url_values[0] if url_values else ''
            raw_url = _text(url_values)
            url = target_url(raw_url, page_url, self.config.type) if raw_url else None
            if title and url:
                hits.append(SearchHit(rank, title, url, _text(self.xpath['snippet_xpath'](item)), self.name))
                if len(hits) == k:
                    break
        if hits:
            return ProviderResult(self.name, 'ok', hits)
        if self.xpath.get('empty_xpath') is not None and self.xpath['empty_xpath'](tree):
            return ProviderResult(self.name, 'empty')
        return None


for _engine in ('google', 'bing', 'yandex', 'xpath'):
    register('search', _engine)(BrowserSearchProvider)


class BrowserSession:
    def __init__(self, config, timeout):
        self.config, self.timeout = config, timeout
        self.driver = None
        self._profile_lock = None
        self.binary = self._find_binary()

    def _find_binary(self):
        if self.config.binary_path is not None:
            if not self.config.binary_path.is_file():
                raise ValueError('browser_binary_missing')
            return str(self.config.binary_path)
        candidates = {
            'firefox': ['/Applications/Firefox.app/Contents/MacOS/firefox', shutil.which('firefox')],
            'chrome': ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', shutil.which('google-chrome')],
            'ungoogled-chromium': ['/Applications/Chromium.app/Contents/MacOS/Chromium', shutil.which('ungoogled-chromium')],
        }[self.config.kind]
        path = next((p for p in candidates if p and Path(p).is_file()), None)
        if path is None:
            raise ValueError('browser_binary_missing')
        return path

    def _prepare_profile(self):
        if self.config.profile_dir is not None:
            return self.config.profile_dir

        import fcntl

        target = Path.home() / 'Library' / 'Application Support' / 'dona-search' / 'browser_profiles' / self.config.kind
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        migration_lock = target.parent / f'.{self.config.kind}-migration.lock'
        with migration_lock.open('a') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if target.exists() and any(target.iterdir()):
                return target
            legacy = Path(__file__).resolve().parents[2] / 'data' / 'browser_profiles' / self.config.kind
            if not legacy.exists():
                return target
            marker = legacy / '.dona-profile'
            if not marker.is_file():
                raise ValueError('legacy_profile_not_owned')
            kind = marker.read_text().strip()
            if kind and kind != self.config.kind:
                raise ValueError('browser_profile_kind_mismatch')
            # Read-lock the source: the migration never writes into the old installation.
            with (legacy / '.dona-lock').open('rb') as source_lock:
                fcntl.flock(source_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                transient = {'.dona-lock', '.parentlock', 'parent.lock', 'lock',
                             'SingletonLock', 'SingletonCookie', 'SingletonSocket'}
                for entry in legacy.rglob('*'):
                    if entry.is_symlink() and not (entry.parent == legacy and entry.name in transient):
                        raise ValueError('legacy_profile_symlink_unsupported')
                def ignore_runtime(directory, names):
                    return set(names) & transient if Path(directory) == legacy else set()
                with tempfile.TemporaryDirectory(prefix=f'.{self.config.kind}-migration-', dir=target.parent) as staging:
                    copy = Path(staging) / 'profile'
                    shutil.copytree(legacy, copy, ignore=ignore_runtime)
                    # Installed profiles can be read-only; only the new copy gains write access.
                    for entry in [copy, *copy.rglob('*')]:
                        permissions = stat.S_IRUSR | stat.S_IWUSR
                        if entry.is_dir():
                            permissions |= stat.S_IXUSR
                        entry.chmod(entry.stat().st_mode | permissions)
                    copy.replace(target)
        return target

    def start(self):
        import fcntl

        from selenium import webdriver
        from selenium.webdriver.common.selenium_manager import SeleniumManager

        if self.driver is not None:
            return
        profile = self._prepare_profile()
        profile.mkdir(parents=True, exist_ok=True)
        marker = profile / '.dona-profile'
        if not marker.exists() and any(profile.iterdir()):
            raise ValueError('browser_profile_not_owned')
        marker.touch()
        self._profile_lock = (profile / '.dona-lock').open('a')
        try:
            fcntl.flock(self._profile_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            kind = marker.read_text().strip()
            if kind and kind != self.config.kind:
                raise ValueError('browser_profile_kind_mismatch')
            marker.write_text(self.config.kind)
            driver_path = self.config.driver_path
            if driver_path is None:
                paths = SeleniumManager().binary_paths([
                    '--browser', 'firefox' if self.config.kind == 'firefox' else 'chrome',
                    '--browser-path', self.binary, '--avoid-browser-download', '--avoid-stats',
                ])
                driver_path = paths['driver_path']
            if self.config.kind == 'firefox':
                from selenium.webdriver.firefox.service import Service

                options = webdriver.FirefoxOptions()
                options.binary_location = self.binary
                options.add_argument('--profile')
                options.add_argument(str(profile))
                if self.config.headless:
                    options.add_argument('-headless')
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    port = sock.getsockname()[1]
                service = Service(str(driver_path), service_args=['--marionette-port', str(port)])
                self.driver = webdriver.Firefox(service=service, options=options)
            else:
                from selenium.webdriver.chrome.service import Service

                options = webdriver.ChromeOptions()
                options.binary_location = self.binary
                options.add_argument(f'--user-data-dir={profile}')
                if self.config.headless:
                    options.add_argument('--headless=new')
                self.driver = webdriver.Chrome(service=Service(str(driver_path)), options=options)
            self.driver.set_page_load_timeout(self.timeout)
        except BaseException:
            self.close()
            raise

    def search(self, provider, query, url, k):
        from selenium.common.exceptions import TimeoutException, WebDriverException
        from selenium.webdriver.support.ui import WebDriverWait
        from urllib3.exceptions import HTTPError

        deadline = time.monotonic() + self.timeout
        navigated = False
        outcome = None
        try:
            self.driver.set_page_load_timeout(self.timeout)
            self.driver.get(url)
            navigated = True
            def inspect(_):
                try:
                    return provider.inspect(self.driver.page_source, self.driver.current_url, query, k)
                except etree.ParserError:
                    return None
            outcome = WebDriverWait(self.driver, max(0, deadline - time.monotonic()), poll_frequency=.25).until(inspect)
        except TimeoutException:
            outcome = ProviderResult(provider.name, 'parse_error' if navigated else 'timeout',
                                     error_code='results_not_recognized' if navigated else 'page_navigation_timeout')
        except (WebDriverException, HTTPError, OSError) as error:
            outcome = ProviderResult(provider.name, 'network_error', error_code=type(error).__name__)
        except Exception as error:
            outcome = ProviderResult(provider.name, 'parse_error', error_code=type(error).__name__)
        finally:
            try:
                self.driver.get('about:blank')
            except Exception as error:
                if outcome is not None:
                    outcome.cleanup_error = type(error).__name__
        return outcome

    def close(self):
        try:
            if self.driver is not None:
                self.driver.quit()
        finally:
            self.driver = None
            if self._profile_lock is not None:
                self._profile_lock.close()
                self._profile_lock = None
