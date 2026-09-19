"""面试复盘：把一场面试记录提炼成经验，沉淀进 experience.md，下一场面试自动生效。

用法：
  python review.py                        # 复盘最近一场
  python review.py logs/session-xxx.md    # 复盘指定一场
"""
import sys
import tomllib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REVIEW_PROMPT = """你是面试复盘教练。下面是一场模拟面试的记录（面试官问题 + AI 生成的答题要点）。
请输出两部分，中间用单独一行 `===EXPERIENCE===` 隔开：

第一部分：本场复盘（Markdown）
- 总体评价 2~3 句（要点覆盖度、结构、有无明显硬伤）
- 逐题点评：每题一行，格式"- 问题关键词 → 点评"

第二部分：更新后的经验清单（Markdown 无序列表）
- 把【旧经验清单】和新本场能提炼的教训合并、去重、改写得更可执行
- 每条一行，面向"下一场面试的我"（例如：回答项目题先报数字结果再讲过程）
- 最多 8 条，按重要性排序；旧经验里已经过时的可以改写

【旧经验清单】
{old}

【本场面试记录】
{log}
"""


def main():
    logs_dir = ROOT / "logs"
    if len(sys.argv) > 1:
        log_file = Path(sys.argv[1])
    else:
        files = sorted(logs_dir.glob("session-*.md"))
        if not files:
            print("logs/ 里还没有面试记录")
            sys.exit(1)
        log_file = files[-1]
    if not log_file.exists():
        print(f"文件不存在: {log_file}")
        sys.exit(1)

    log_text = log_file.read_text(encoding="utf-8")
    if "**要点**" not in log_text:
        print(f"{log_file.name} 里没有带回答的记录（可能当时没配 API Key），无法复盘")
        sys.exit(1)

    exp_file = ROOT / "experience.md"
    old_exp = exp_file.read_text(encoding="utf-8") if exp_file.exists() else "（暂无，这是第一场）"

    cfg = tomllib.load(open(ROOT / "config.toml", "rb"))["llm"]
    if not cfg.get("api_key"):
        print("未配置 API Key，无法复盘")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=60, max_retries=1)
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=[{"role": "user", "content": REVIEW_PROMPT.format(
            old=old_exp[:1000], log=log_text[:6000])}],
        temperature=0.3,
        max_tokens=1200,
    )
    content = (resp.choices[0].message.content or "").strip()
    if "===EXPERIENCE===" not in content:
        sys.exit("模型输出格式异常，原始内容：\n" + content)

    review, experience = content.split("===EXPERIENCE===", 1)

    exp_dir = ROOT / "experience"
    exp_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(exp_dir / "reviews.md", "a", encoding="utf-8") as f:
        f.write(f"\n## 复盘 {stamp}（{log_file.name}）\n\n{review.strip()}\n")
    exp_file.write_text(experience.strip() + "\n", encoding="utf-8")

    print(f"复盘完成（{log_file.name}）")
    print("- 逐题点评已追加: experience/reviews.md")
    print("- 经验清单已更新: experience.md（下一场面试自动生效）")
    print(f"- 目前共积累 {len(experience.strip().splitlines())} 行经验")


if __name__ == "__main__":
    main()
