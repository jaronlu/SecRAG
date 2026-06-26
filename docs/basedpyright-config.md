# Basedpyright 虚拟环境配置

## 问题

basedpyright 无法解析 `.venv` 中的第三方包导入。

## 方案

`pyproject.toml` 中配置 `[tool.basedpyright]`：

```toml
[tool.basedpyright]
venvPath = "."
venv = ".venv"
```

`venvPath` + `venv` 拼接得到 `./.venv`，basedpyright 从 `pyproject.toml` 所在目录解析。

## 注意

`python.pythonPath` 是编辑器 LSP（Language Server Protocol）选项——编辑器通过 LSP 将代码内容发送给 basedpyright，basedpyright 分析后返回诊断结果。LSP 配置放在编辑器的 `settings.json` 中，与 `pyproject.toml` 是两套不同的配置入口。
