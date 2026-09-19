"""LLM 答题要点：把面试官的问题 + 候选人画像交给大模型，流式返回要点。"""
from __future__ import annotations

from openai import OpenAI

SYSTEM_PROMPT = """\
你是一个实时面试辅助助手。面试官刚问了一个问题，请给出简明扼要的答题要点。

要求：
1. 用 3~5 条要点，每条不超过 30 字，第一条给核心结论；
2. 只输出要点本身（用 - 开头），不要客套话、不要复述问题；
3. 回答尽量结合候选人的背景和项目经验；
4. 是开放题/场景题时，优先给出答题框架（比如"先定义问题 → 给方案 → 说权衡"）。

【候选人背景】
{persona}
"""

FULL_ANSWER_PROMPT = """\
你是一个实时面试辅助助手。面试官刚问了一个问题，请给出一个可以直接口述的完整回答。

要求：
1. 开头第一句直接给结论或核心观点；
2. 结合候选人背景用 STAR 结构展开（情境一句、任务一句、行动两三句、结果一句）；
3. 全文不超过 200 字，用口语化短句，方便照着说；
4. 不要客套话、不要"作为候选人"之类的自称。

【候选人背景】
{persona}
"""

_MODE_PROMPTS = {"brief": SYSTEM_PROMPT, "full": FULL_ANSWER_PROMPT}


class Advisor:
    def __init__(self, api_key: str, base_url: str, model: str, persona: str, mode: str = "brief"):
        self.model = model
        # 面试工具必须快失败：网络半死不活时最多等 30 秒，不重试拖时间
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=30, max_retries=1)
        self.system = _MODE_PROMPTS.get(mode, SYSTEM_PROMPT).format(persona=persona)

    def stream_answer(self, question: str):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": question},
            ],
            stream=True,
            temperature=0.3,
            max_tokens=2048,  # 思考型模型（如 deepseek-v4-flash）的思考过程也计入配额，要留够
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
