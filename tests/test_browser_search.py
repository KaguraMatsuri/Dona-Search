from pathlib import Path

import pytest

from dona_search.config import BrowserConfig, BrowserProviderConfig, ExtractionRules
from dona_search.search.browser import BrowserSearchProvider, BrowserSession, target_url


@pytest.fixture
def browser_config(tmp_path):
    binary = tmp_path / 'fake-browser'
    binary.write_bytes(b'fixture')
    return lambda **kwargs: BrowserConfig(**({'kind': 'firefox', 'binary_path': binary} | kwargs))


def provider(engine, **kwargs):
    return BrowserSearchProvider(BrowserProviderConfig(type=f'search_{engine}', name=engine, **kwargs), None)


@pytest.mark.parametrize('engine,body', [
    ('google', '<div class="MjjYud"><a href="https://sqlite.org/docs.html"><h3>Documentation - SQLite</h3></a><div class="VwiC3b">Original snippet</div></div>'),
    ('bing', '<li class="b_algo"><h2><a href="https://sqlite.org/docs.html">Documentation - SQLite</a></h2><div class="b_caption"><p>Original snippet</p></div></li>'),
    ('yandex', '<li class="serp-item"><h2><a href="https://sqlite.org/docs.html">Documentation - SQLite</a></h2><div class="OrganicText">Original snippet</div></li>'),
])
def test_engine_rules_extract_original_fields(engine, body):
    result = provider(engine).inspect(body, f'https://{engine}.com/search', 'SQLite', 20)
    assert result.status == 'ok'
    assert (result.hits[0].title, result.hits[0].url, result.hits[0].snippet) == (
        'Documentation - SQLite', 'https://sqlite.org/docs.html', 'Original snippet')


def test_custom_engine_can_be_added_only_by_configuration():
    rules = ExtractionRules(item_xpath='//article', title_xpath='./h2', url_xpath='./a/@href',
                            snippet_xpath='./p', query_xpath='//input/@value', empty_xpath="//*[@id='empty']")
    engine = provider('xpath', search_url='https://custom.test/find?fixed=yes', query_param='term', rules=rules)
    class Browser:
        def search(self, provider, query, url, k):
            from urllib.parse import parse_qs, urlsplit
            assert parse_qs(urlsplit(url).query) == {'fixed':['yes'], 'term':['A&B + 中文']}
            return provider.inspect('<input value="A&amp;B + 中文"><article><h2>原始标题</h2><a href="/doc"></a><p>原始片段</p></article>', url, query, k)
    engine.browser = Browser()
    result = engine.search('A&B + 中文', 20)
    assert result.hits[0].url == 'https://custom.test/doc'
    assert result.hits[0].snippet == '原始片段'
    assert engine.inspect('<div id="empty"></div>', 'https://custom.test/find', 'q', 20).status == 'empty'


def test_query_mismatch_challenges_and_unrecognized_page_are_not_success():
    engine = provider('bing')
    assert engine.inspect('<input name="q" value="unrelated">', 'https://bing.com/search', 'SQLite', 20).status == 'query_mismatch'
    assert engine.inspect('<html></html>', 'https://bing.com/search', 'SQLite', 20) is None
    assert engine.inspect('<html></html>', 'https://google.com/sorry/index', 'SQLite', 20).error_code == 'captcha_required'
    assert engine.inspect('<html></html>', 'https://consent.google.com/', 'SQLite', 20).error_code == 'consent_required'
    assert provider('yandex').inspect('<html></html>', 'https://yandex.com/showcaptcha', 'q', 20).status == 'blocked'


def test_redirect_targets_and_unsafe_links():
    assert target_url('/url?q=https%3A%2F%2Fsqlite.org%2F', 'https://www.google.com/search', 'search_google') == 'https://sqlite.org/'
    assert target_url('javascript:alert(1)', 'https://bing.com', 'search_bing') is None
    assert target_url('https://user:password@example.test', 'https://bing.com', 'search_bing') is None
    assert target_url('/ck/a?u=bad', 'https://bing.com', 'search_bing') is None


def test_missing_selected_browser_does_not_fallback(monkeypatch):
    monkeypatch.setattr('dona_search.search.browser.shutil.which', lambda _: None)
    monkeypatch.setattr(Path, 'is_file', lambda _: False)
    with pytest.raises(ValueError, match='browser_binary_missing'):
        BrowserSession(BrowserConfig(kind='ungoogled-chromium'), 20)


def test_personal_or_unknown_profile_is_rejected(tmp_path, browser_config):
    (tmp_path / 'prefs.js').write_text('personal fixture')
    session = BrowserSession(browser_config(profile_dir=tmp_path), 20)
    with pytest.raises(ValueError, match='browser_profile_not_owned'):
        session.start()
    assert (tmp_path / 'prefs.js').read_text() == 'personal fixture'


def test_profile_lock_lifecycle_and_firefox_in_place_options(tmp_path, monkeypatch, browser_config):
    instances = []
    class Driver:
        def __init__(self, service, options):
            self.args = options.arguments
            self.closed = False
            instances.append(self)
        def set_page_load_timeout(self, value): pass
        def quit(self): self.closed = True
    monkeypatch.setattr('selenium.webdriver.Firefox', Driver)
    cfg = browser_config(driver_path=tmp_path/'driver', profile_dir=tmp_path/'profile')
    first = BrowserSession(cfg, 20)
    second = BrowserSession(cfg, 20)
    first.start()
    first.start()
    assert len(instances) == 1
    with pytest.raises(BlockingIOError): second.start()
    assert instances[0].args == ['--profile', str(cfg.profile_dir), '-headless']
    first.close()
    second.start()
    second.close()
    assert all(driver.closed for driver in instances)
    assert (cfg.profile_dir/'.dona-profile').exists()


def test_profile_cannot_be_reused_by_different_browser_kind(tmp_path, browser_config):
    (tmp_path / '.dona-profile').write_text('chrome')
    session = BrowserSession(browser_config(profile_dir=tmp_path), 20)
    with pytest.raises(ValueError, match='browser_profile_kind_mismatch'):
        session.start()
    assert (tmp_path / '.dona-profile').read_text() == 'chrome'


def test_navigation_and_dom_wait_share_one_timeout_budget(monkeypatch, browser_config):
    times = iter([0, 7])
    monkeypatch.setattr('dona_search.search.browser.time.monotonic', lambda: next(times))
    observed = []
    class Driver:
        page_source = '<html></html>'
        current_url = 'https://yandex.com/showcaptcha'
        def set_page_load_timeout(self, value): observed.append(('navigation_timeout', value))
        def get(self, url): observed.append(('get', url))
    class Wait:
        def __init__(self, driver, timeout, **kwargs): observed.append(('dom_timeout', timeout))
        def until(self, inspect): return inspect(None)
    monkeypatch.setattr('selenium.webdriver.support.ui.WebDriverWait', Wait)
    session = BrowserSession(browser_config(),20)
    session.driver = Driver()
    result = session.search(provider('yandex'), 'q', 'https://yandex.com/search?text=q', 20)
    assert result.status == 'blocked'
    assert ('dom_timeout', 13) in observed
    assert observed[-1] == ('get', 'about:blank')


@pytest.mark.parametrize('navigation', [True, False])
def test_browser_distinguishes_navigation_timeout_from_unrecognized_results(monkeypatch, navigation, browser_config):
    from selenium.common.exceptions import TimeoutException
    calls = []
    class Driver:
        def set_page_load_timeout(self, value): pass
        def get(self, url):
            calls.append(url)
            if navigation and url != 'about:blank': raise TimeoutException()
    class Wait:
        def __init__(self, *args, **kwargs): pass
        def until(self, callback): raise TimeoutException()
    monkeypatch.setattr('selenium.webdriver.support.ui.WebDriverWait', Wait)
    session = BrowserSession(browser_config(), 20)
    session.driver = Driver()
    result = session.search(provider('bing'), 'q', 'https://bing.com/search', 20)
    assert result.status == ('timeout' if navigation else 'parse_error')
    assert calls[-1] == 'about:blank'


def test_google_nested_snippet_and_duplicate_echo_are_not_duplicated():
    source = ('<textarea name="q">SQLite</textarea><input name="q" value="SQLite">'
              '<div class="MjjYud"><a href="https://sqlite.org"><h3>SQLite</h3></a>'
              '<div class="IsZvec"><div class="VwiC3b">Original snippet</div></div></div>')
    result = provider('google').inspect(source, 'https://google.com/search', 'SQLite', 20)
    assert result.status == 'ok' and result.hits[0].snippet == 'Original snippet'


def test_field_xpath_is_scoped_to_each_result_even_inside_functions():
    rules = ExtractionRules(item_xpath='//article', title_xpath='normalize-space(//h2)',
                            url_xpath='./a/@href', snippet_xpath='string(//p)')
    engine = provider('xpath', search_url='https://custom.test/search', rules=rules)
    source = ('<article><h2>SQLite docs</h2><a href="https://sqlite.org/docs.html"></a><p>SQLite original</p></article>'
              '<article><h2>Minecraft wiki</h2><a href="https://minecraft.wiki/"></a><p>Minecraft original</p></article>')
    result = engine.inspect(source, 'https://custom.test/search', 'q', 20)
    assert [(h.title, h.url, h.snippet) for h in result.hits] == [
        ('SQLite docs', 'https://sqlite.org/docs.html', 'SQLite original'),
        ('Minecraft wiki', 'https://minecraft.wiki/', 'Minecraft original'),
    ]


@pytest.mark.parametrize('expression', [' /html/h2', '\n//h2'])
def test_leading_whitespace_does_not_bypass_relative_xpath_validation(expression):
    with pytest.raises(ValueError, match='result_xpath_must_be_relative'):
        ExtractionRules(item_xpath='//article', title_xpath=expression, url_xpath='./a/@href', snippet_xpath='./p')


@pytest.mark.parametrize('error_type', ['read_timeout', 'webdriver', 'os_error'])
def test_page_cleanup_failure_preserves_successful_results_and_reports_error(browser_config, error_type):
    from selenium.common.exceptions import WebDriverException
    from urllib3.exceptions import ReadTimeoutError
    from dona_search.search.scheduler import SearchScheduler, merge_tasks
    from dona_search.interfaces import SearchTask, TaskResult
    errors = {'read_timeout': ReadTimeoutError(None, 'fixture', 'cleanup'),
              'webdriver': WebDriverException('cleanup'), 'os_error': ConnectionResetError('cleanup')}
    class Driver:
        page_source = '<li class="b_algo"><h2><a href="https://sqlite.org/">SQLite</a></h2><div class="b_caption"><p>Original</p></div></li>'
        current_url = 'https://bing.com/search?q=SQLite'
        def set_page_load_timeout(self, value): pass
        def get(self, url):
            if url == 'about:blank': raise errors[error_type]
    session = BrowserSession(browser_config(), 1)
    session.driver = Driver()
    engine = provider('bing')
    engine.browser = session
    outcome = SearchScheduler._execute(engine, 'SQLite')
    assert outcome.status == 'ok' and outcome.error_code is None
    assert len(outcome.hits) == 1 and outcome.cleanup_error == type(errors[error_type]).__name__
    outcome.hits[0].origins = [{'provider':'bing'}]
    hits, providers = merge_tasks([TaskResult(SearchTask(0, 'bing', 'SQLite', 0, 20), outcome)], ['bing'])
    assert len(hits) == 1 and providers[0].cleanup_error == outcome.cleanup_error


def test_navigation_transport_error_is_not_reported_as_parse_failure(browser_config):
    from urllib3.exceptions import ReadTimeoutError
    class Driver:
        def set_page_load_timeout(self, value): pass
        def get(self, url):
            if url != 'about:blank': raise ReadTimeoutError(None, url, 'navigation')
    session = BrowserSession(browser_config(), 1)
    session.driver = Driver()
    outcome = session.search(provider('bing'), 'q', 'https://bing.com/search', 20)
    assert outcome.status == 'network_error' and outcome.error_code == 'ReadTimeoutError'


def test_cleanup_failure_does_not_replace_original_parse_failure(browser_config):
    from urllib3.exceptions import ReadTimeoutError
    class Driver:
        page_source = '<html></html>'
        current_url = 'https://custom.test/search'
        def set_page_load_timeout(self, value): pass
        def get(self, url):
            if url == 'about:blank': raise ReadTimeoutError(None, url, 'cleanup')
    rules = ExtractionRules(item_xpath='true()', title_xpath='./h2', url_xpath='./a/@href', snippet_xpath='./p')
    engine = provider('xpath', search_url='https://custom.test/search', rules=rules)
    session = BrowserSession(browser_config(), 1)
    session.driver = Driver()
    outcome = session.search(engine, 'q', 'https://custom.test/search', 20)
    assert outcome.status == 'parse_error' and outcome.error_code == 'ValueError'
    assert outcome.cleanup_error == 'ReadTimeoutError'


@pytest.fixture
def profile_paths(tmp_path, monkeypatch):
    import dona_search.search.browser as module
    home = tmp_path / 'home'
    install = tmp_path / 'install'
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: home))
    monkeypatch.setattr(module, '__file__', str(install / 'dona_search/search/browser.py'))
    legacy = install / 'data/browser_profiles/firefox'
    target = home / 'Library/Application Support/dona-search/browser_profiles/firefox'
    return legacy, target


def owned_legacy(path):
    path.mkdir(parents=True)
    (path / '.dona-profile').write_text('firefox')
    (path / '.dona-lock').touch()
    (path / 'cookies.sqlite').write_bytes(b'cookie fixture')
    (path / 'prefs.js').write_text('prefs fixture')


def test_default_profile_migrates_owned_session_once_and_keeps_source(profile_paths, browser_config):
    legacy, target = profile_paths
    owned_legacy(legacy)
    (legacy / '.parentlock').symlink_to('inactive-runtime-lock')
    before = {p.name: p.read_bytes() for p in legacy.iterdir() if p.is_file()}
    session = BrowserSession(browser_config(), 20)
    assert session._prepare_profile() == target
    assert (target / 'cookies.sqlite').read_bytes() == b'cookie fixture'
    assert (target / 'prefs.js').read_text() == 'prefs fixture'
    assert not (target / '.parentlock').exists()
    assert not (target / '.dona-lock').exists()
    assert before == {p.name: p.read_bytes() for p in legacy.iterdir() if p.is_file()}
    (target / 'cookies.sqlite').write_bytes(b'new session cookie')
    (legacy / 'cookies.sqlite').write_bytes(b'old session changed')
    assert session._prepare_profile() == target
    assert (target / 'cookies.sqlite').read_bytes() == b'new session cookie'


def test_explicit_profile_does_not_migrate_or_create_default(profile_paths, browser_config, tmp_path):
    legacy, target = profile_paths
    owned_legacy(legacy)
    explicit = tmp_path / 'explicit'
    session = BrowserSession(browser_config(profile_dir=explicit), 20)
    assert session._prepare_profile() == explicit
    assert not target.parent.exists()
    assert not explicit.exists()


def test_no_legacy_profile_uses_user_directory_without_writing_install(profile_paths, browser_config):
    legacy, target = profile_paths
    session = BrowserSession(browser_config(), 20)
    assert session._prepare_profile() == target
    assert target.parent.is_dir()
    assert not legacy.parents[2].exists()


def test_active_legacy_profile_is_not_copied(profile_paths, browser_config):
    import fcntl
    legacy, target = profile_paths
    owned_legacy(legacy)
    with (legacy / '.dona-lock').open('a') as active:
        fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
        session = BrowserSession(browser_config(), 20)
        with pytest.raises(BlockingIOError):
            session._prepare_profile()
        assert not target.exists()
    assert session._prepare_profile() == target


def test_copy_failure_leaves_source_and_target_unmodified(profile_paths, browser_config, monkeypatch):
    legacy, target = profile_paths
    owned_legacy(legacy)
    def fail(source, destination, **kwargs):
        destination.mkdir()
        (destination / 'partial-file').write_bytes(b'partial')
        raise OSError('fixture copy failure')
    monkeypatch.setattr('dona_search.search.browser.shutil.copytree', fail)
    with pytest.raises(OSError, match='fixture copy failure'):
        BrowserSession(browser_config(), 20)._prepare_profile()
    assert not target.exists()
    assert (legacy / 'cookies.sqlite').read_bytes() == b'cookie fixture'
    assert not list(target.parent.glob('.firefox-migration-*'))


@pytest.mark.parametrize('kind', [None, 'chrome'])
def test_unowned_or_wrong_kind_legacy_is_not_migrated(profile_paths, browser_config, kind):
    legacy, target = profile_paths
    legacy.mkdir(parents=True)
    (legacy / 'cookies.sqlite').write_bytes(b'untouched')
    if kind:
        (legacy / '.dona-profile').write_text(kind)
    with pytest.raises(ValueError, match='legacy_profile_not_owned' if kind is None else 'browser_profile_kind_mismatch'):
        BrowserSession(browser_config(), 20)._prepare_profile()
    assert not target.exists() and (legacy / 'cookies.sqlite').read_bytes() == b'untouched'


def test_symlink_outside_legacy_is_not_followed(profile_paths, browser_config, tmp_path):
    legacy, target = profile_paths
    owned_legacy(legacy)
    private = tmp_path / 'unrelated'
    private.write_bytes(b'unrelated fixture')
    (legacy / 'external-link').symlink_to(private)
    with pytest.raises(ValueError, match='legacy_profile_symlink_unsupported'):
        BrowserSession(browser_config(), 20)._prepare_profile()
    assert not target.exists() and private.read_bytes() == b'unrelated fixture'


def test_read_only_legacy_is_unchanged_but_new_copy_is_writable(profile_paths, browser_config):
    import stat
    legacy, target = profile_paths
    owned_legacy(legacy)
    for p in legacy.iterdir(): p.chmod(0o400)
    legacy.chmod(0o500)
    try:
        BrowserSession(browser_config(), 20)._prepare_profile()
        assert (legacy / 'cookies.sqlite').stat().st_mode & stat.S_IWUSR == 0
        assert legacy.stat().st_mode & stat.S_IWUSR == 0
        assert (target / 'cookies.sqlite').stat().st_mode & stat.S_IWUSR
        assert target.stat().st_mode & stat.S_IWUSR
    finally:
        legacy.chmod(0o700)
        for p in legacy.iterdir(): p.chmod(0o600)


def test_empty_existing_target_can_receive_migration(profile_paths, browser_config):
    legacy, target = profile_paths
    owned_legacy(legacy)
    target.mkdir(parents=True)
    assert BrowserSession(browser_config(), 20)._prepare_profile() == target
    assert (target / 'cookies.sqlite').read_bytes() == b'cookie fixture'
