from importlib.resources import files
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HTTP_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
}


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Literal['search_so360', 'search_mwmbl']
    name: str = Field(min_length=1)
    enabled: bool = True
    top_k: int = Field(default=20, ge=1, le=20, strict=True)
    search_url: str | None = None
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode='after')
    def valid_params(self):
        if not self.name.strip():
            raise ValueError('invalid_provider_name')
        reserved = 'q' if self.type == 'search_so360' else 's'
        if reserved in self.params:
            raise ValueError('query_parameter_reserved')
        return self

    @field_validator('search_url')
    @classmethod
    def valid_url(cls, value):
        if value is not None:
            parsed = urlsplit(value)
            if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
                raise ValueError('invalid_search_url')
        return value


class RankingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = 'google/gemma-4-E4B-it'
    revision: str = 'ee0ef6023621cff504d758262d4e04895a5af4a2'
    model_path: Path | None = None
    adapter_path: Path | None = None
    max_prompt_tokens: int = Field(default=8192, ge=256, le=8192, strict=True)


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal[2] = 2
    providers: list[ProviderConfig] = Field(default_factory=lambda: [
        ProviderConfig(type='search_so360', name='360'),
        ProviderConfig(type='search_mwmbl', name='mwmbl'),
    ])
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    display_limit: int = Field(default=10, ge=1, le=10, strict=True)
    timeout: float = Field(default=20, gt=0, le=60)

    @model_validator(mode='after')
    def valid_providers(self):
        names = [p.name for p in self.providers]
        active = [p for p in self.providers if p.enabled]
        if len(names) != len(set(names)):
            raise ValueError('duplicate_provider_name')
        if not active:
            raise ValueError('no_enabled_providers')
        if sum(p.top_k for p in active) > 40:
            raise ValueError('candidate_limit_exceeded')
        return self


class ExtractionRules(BaseModel):
    model_config = ConfigDict(extra='forbid')
    item_xpath: str = Field(min_length=1)
    title_xpath: str = Field(min_length=1)
    url_xpath: str = Field(min_length=1)
    snippet_xpath: str = Field(min_length=1)
    query_xpath: str | None = None
    empty_xpath: str | None = None
    blocked_xpath: str | None = None

    @model_validator(mode='after')
    def compile_rules(self):
        from lxml import etree

        for value in self.model_dump().values():
            if value is not None:
                try:
                    etree.XPath(value)
                except etree.XPathError as error:
                    raise ValueError('invalid_extraction_xpath') from error
        for value in (self.title_xpath, self.url_xpath, self.snippet_xpath):
            if value.lstrip().startswith('/'):
                raise ValueError('result_xpath_must_be_relative')
        return self


class BrowserProviderConfig(ProviderConfig):
    type: Literal['search_google', 'search_bing', 'search_yandex', 'search_xpath',
                  'search_so360', 'search_mwmbl']
    query_param: str | None = Field(default=None, min_length=1)
    rules: ExtractionRules | None = None

    @model_validator(mode='after')
    def valid_params(self):
        if not self.name.strip():
            raise ValueError('invalid_provider_name')
        self.query_param = self.query_param or {
            'search_yandex': 'text', 'search_mwmbl': 's',
        }.get(self.type, 'q')
        if self.query_param in self.params:
            raise ValueError('query_parameter_reserved')
        if self.type == 'search_xpath' and (self.search_url is None or self.rules is None):
            raise ValueError('custom_engine_requires_url_and_rules')
        return self


class BrowserConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['ungoogled-chromium', 'firefox', 'chrome'] = 'ungoogled-chromium'
    binary_path: Path | None = None
    driver_path: Path | None = None
    profile_dir: Path | None = None
    headless: bool = True


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal[3] = 3
    providers: list[BrowserProviderConfig] = Field(default_factory=lambda: [
        BrowserProviderConfig(type='search_google', name='google'),
        BrowserProviderConfig(type='search_bing', name='bing'),
        BrowserProviderConfig(type='search_yandex', name='yandex'),
    ])
    query_model: RankingConfig = Field(default_factory=lambda: RankingConfig(max_prompt_tokens=2048))
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    max_queries: int = Field(default=3, ge=1, le=3, strict=True)
    display_limit: int = Field(default=10, ge=1, le=10, strict=True)
    timeout: float = Field(default=20, gt=0, le=60)

    @model_validator(mode='after')
    def valid_providers(self):
        names = [p.name for p in self.providers]
        if len(names) != len(set(names)):
            raise ValueError('duplicate_provider_name')
        if not any(p.enabled for p in self.providers):
            raise ValueError('no_enabled_providers')
        return self


MODEL_PRESETS = (
    ('Qwen/Qwen3-4B', '1cfa9a7208912126459214e8b04321603b3df60c'),
    ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218'),
    ('google/gemma-4-E2B-it', '3e22461f65e89153144f8adb70e3b8c2cc9845a7'),
    ('google/gemma-4-E4B-it', 'ee0ef6023621cff504d758262d4e04895a5af4a2'),
)


class ArtifactSource(BaseModel):
    model_config = ConfigDict(extra='forbid')
    repo: str | None = Field(default=None, min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    path: Path | None = None

    @model_validator(mode='after')
    def one_source(self):
        if (self.repo is None) == (self.path is None):
            raise ValueError('artifact_requires_one_source')
        if self.path is not None and self.revision is not None:
            raise ValueError('local_artifact_has_no_revision')
        return self


class ModelSelection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = Field(default=MODEL_PRESETS[3][0], min_length=1)
    revision: str | None = None
    weights: ArtifactSource | None = None
    adapter: ArtifactSource | None = None
    adapter_base_fingerprint: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    max_prompt_tokens: int = Field(default=8192, ge=256, le=8192, strict=True)

    @model_validator(mode='after')
    def pin_preset(self):
        if self.revision is None:
            self.revision = dict(MODEL_PRESETS).get(self.model)
        if self.adapter is None and self.adapter_base_fingerprint is not None:
            raise ValueError('adapter_binding_without_adapter')
        return self

    @property
    def model_path(self):
        return self.weights.path if self.weights is not None else None

    @property
    def adapter_path(self):
        return self.adapter.path if self.adapter is not None else None

    @property
    def allow_external_adapters(self):
        return True


class PipelineConfigV4(PipelineConfig):
    schema_version: Literal[4] = 4
    query_model: ModelSelection = Field(default_factory=lambda: ModelSelection(max_prompt_tokens=2048))
    ranking: ModelSelection = Field(default_factory=ModelSelection)
    memory_budget_bytes: int | None = Field(default=None, gt=0, strict=True)


def user_config_path():
    return Path.home() / 'Library' / 'Application Support' / 'dona-search' / 'config.yaml'


def default_config():
    return yaml.safe_load(files('dona_search').joinpath('resources/default.yaml').read_text())


def selection_config(config):
    if isinstance(config, PipelineConfigV4):
        return config
    if not isinstance(config, PipelineConfig):
        raise TypeError('console_requires_schema_v3_or_v4')
    data = config.model_dump(mode='json')
    data['schema_version'] = 4
    for role in ('query_model', 'ranking'):
        old = data[role]
        data[role] = {'model': old['model'], 'revision': old['revision'], 'max_prompt_tokens': old['max_prompt_tokens'],
                      'weights': {'path': old['model_path']} if old['model_path'] else None,
                      'adapter': {'path': old['adapter_path']} if old['adapter_path'] else None}
    return PipelineConfigV4.model_validate(data)


def read_config(path: str | Path | None):
    if isinstance(path, PipelineConfig):
        return path.model_dump(mode='python'), Path.cwd()
    implicit = path is None
    if implicit:
        path = user_config_path()
        if not path.exists():
            return default_config(), Path.cwd()
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError('invalid_config_document')
    if implicit and raw.get('schema_version') == 3:
        raw = selection_config(validate_config(raw, path.parent, require_model_paths=False)).model_dump(mode='python')
    return raw, path.parent


def validate_config(raw: dict, base: Path, *, require_model_paths=True,
                    required_models=('ranking', 'query_model')):
    version = raw.get('schema_version')
    config_type = {3: PipelineConfig, 4: PipelineConfigV4}.get(version, SearchConfig)
    config = config_type.model_validate(raw)
    models = {'ranking': config.ranking}
    if isinstance(config, PipelineConfig):
        models['query_model'] = config.query_model
        for name in ('binary_path', 'driver_path', 'profile_dir'):
            value = getattr(config.browser, name)
            if value is not None:
                setattr(config.browser, name, (base / value.expanduser()).resolve())
    for name, model in models.items():
        if isinstance(model, ModelSelection):
            model_directory = base / Path(model.model).expanduser()
            if model.model.startswith(('/', '.', '~')) or model_directory.is_dir():
                model.model = str(model_directory.resolve())
            for source in (model.weights, model.adapter):
                if source is not None and source.path is not None:
                    source.path = (base / source.path.expanduser()).resolve()
                    if require_model_paths and name in required_models and not source.path.is_dir():
                        raise ValueError('artifact_path_missing')
            continue
        if model.adapter_path is not None:
            model.adapter_path = (base / model.adapter_path.expanduser()).resolve()
            if require_model_paths and name in required_models and not model.adapter_path.is_dir():
                raise ValueError('adapter_path_missing')
        if model.model_path is not None:
            model.model_path = (base / model.model_path.expanduser()).resolve()
            if require_model_paths and name in required_models and not model.model_path.is_dir():
                raise ValueError('model_path_missing')
    return config
