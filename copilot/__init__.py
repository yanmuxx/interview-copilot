"""面试 Copilot 核心包。首次运行时自动从 example 模板补齐个人配置文件。"""
import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def bootstrap_user_files():
    for name, example in (
        ("config.toml", "config.example.toml"),
        ("persona.md", "persona.example.md"),
    ):
        target = _PROJECT_ROOT / name
        source = _PROJECT_ROOT / example
        if not target.exists() and source.exists():
            shutil.copy2(source, target)


bootstrap_user_files()
