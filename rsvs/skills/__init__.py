"""技能定义：来自 go2_eval_combo_gym_policy 的高动态动作（不含普通行走）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SkillDef:
    key: str
    label: str
    description: str


# 用户要求：简单行走类不要；保留特技 / 高动态
SKILLS: List[SkillDef] = [
    SkillDef(
        key="handstand",
        label="前脚直立",
        description="前腿支撑倒立，用于展示平衡与高动态支撑技能。",
    ),
    SkillDef(
        key="legstand_cycle",
        label="后腿站立循环",
        description="后腿直立升起 → 保持 → 回落。",
    ),
    SkillDef(
        key="backflip",
        label="后空翻",
        description="单次后空翻。",
    ),
    SkillDef(
        key="spring_jump",
        label="弹跳",
        description="向前弹跳一次。",
    ),
    SkillDef(
        key="backflip_double",
        label="连续后空翻",
        description="双次后空翻收尾动作。",
    ),
]

SKILL_BY_KEY = {s.key: s for s in SKILLS}
