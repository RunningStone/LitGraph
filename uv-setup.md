# LitGraph - uv 虚拟环境配置

## 快速开始

```bash
# 1. 安装 uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 创建虚拟环境并安装依赖
cd /Users/shipan/Documents/workspace_automate_life/literature_graph/LitGraph
uv venv
source .venv/bin/activate

# 3. 安装项目（开发模式）
uv pip install -e ".[dev]"
```

## 配置文件说明

- `.python-version` - 指定 Python 3.9
- `pyproject.toml` - 包含 `[tool.uv]` 配置

## 日常使用

```bash
# 激活虚拟环境
source .venv/bin/activate

# 运行 CLI
litgraph --help

# 运行测试
pytest tests/dummy/ -v

# 退出虚拟环境
deactivate
```

## 与全局 miniconda 环境的区别

| 项目 | uv venv | miniconda (全局) |
|------|---------|-----------------|
| 路径 | `.venv/` | `/opt/miniconda3/` |
| 隔离 | 项目级隔离 | 全局共享 |
| 依赖冲突 | 无 | 可能有 |
| 推荐 | 开发时使用 | 不推荐 |

## 注意事项

1. **pandas冲突**: 全局环境中的pandas可能被其他项目破坏，使用uv venv可避免
2. **激活检查**: 使用 `which python` 确认是否在正确的虚拟环境中
3. **IDE配置**: VS Code等IDE需要选择`.venv/bin/python`作为解释器
