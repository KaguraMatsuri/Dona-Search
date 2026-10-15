# 配置与使用

先按 README 安装环境和浏览器，再启动 `.venv/bin/python -m dona_search` 完成两模型准备。SDK 和单次查询依次读取显式配置、上次保存配置、内置默认配置。

## 最小 v4 配置

```yaml
schema_version: 4
providers:
  - type: search_bing
    name: bing
    enabled: true
    top_k: 20
browser:
  kind: ungoogled-chromium
  headless: true
query_model:
  model: google/gemma-4-E4B-it
  max_prompt_tokens: 2048
ranking:
  model: google/gemma-4-E4B-it
  max_prompt_tokens: 8192
max_queries: 3
display_limit: 10
timeout: 20
memory_budget_bytes: null
```

预设模型在省略 revision 时使用代码中的固定提交。`max_queries` 包含原始查询；默认原始查询加最多两条模型查询。`memory_budget_bytes: null` 使用 Metal 推荐预算。

`browser.kind` 可选 `ungoogled-chromium`、`firefox`、`chrome`。可额外指定 `binary_path`、`driver_path` 和项目专用的 `profile_dir`。不要使用个人日常浏览器配置目录；既有非专用目录会被拒绝。

## 模型、完整权重与适配器

```yaml
query_model:
  model: ./models/query-complete
  weights:
    path: ./models/query-complete
  max_prompt_tokens: 2048
ranking:
  model: google/gemma-4-E4B-it
  weights:
    repo: organization/compatible-complete-mlx-weights
    revision: replace-with-a-revision
  max_prompt_tokens: 8192
```

这只是字段示例，需要替换仓库与路径。配置文件中的相对路径以配置文件所在目录为基准。`weights` 是完整 MLX bundle，不是通用跨模型权重；基座架构和大小必须匹配。自定义完整权重不会自动改精度。

可选 `adapter` 也使用 `path` 或 `repo` 加 revision。框架检查适配器任务与精确基座绑定。外部适配器需要可验证的基座指纹或本地基座绑定；`adapter_base_fingerprint` 对应实际加载的完整权重，不能随意填写。内置默认权重会按 8-bit 准备；转换超预算时停止。

## 自定义搜索引擎

```yaml
providers:
  - type: search_xpath
    name: custom
    search_url: https://search.example.com/search
    query_param: q
    top_k: 20
    rules:
      item_xpath: "//li[contains(@class, 'result')]"
      title_xpath: ".//h2"
      url_xpath: ".//h2/a/@href"
      snippet_xpath: ".//p"
      query_xpath: "//input[@name='q']/@value"
      empty_xpath: "//*[@id='no-results']"
      blocked_xpath: "//*[@id='captcha']"
```

替换 URL 和规则后使用。标题、链接和片段 XPath 必须相对于当前结果；`query_xpath`、`empty_xpath`、`blocked_xpath` 在整页执行。内置浏览器预设为 Google、Bing、Yandex；另有 360 和 Mwmbl HTTP 适配器。页面结构变化可能需要更新规则。

## 查询与状态

```bash
.venv/bin/python -m dona_search --config my-config.yaml 'SQLite 事务'
.venv/bin/python -m dona_search --config my-config.yaml --search-only 'SQLite 事务'
```

单次命令的状态行为：模型失败或全部引擎技术失败返回非零退出码；合法空结果或部分引擎成功返回 0。状态行是 JSON，结果按标题、URL、摘要显示。截图或快照中的未标注候选不是训练真值。
