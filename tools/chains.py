"""Tool Chains — 预定义的工具组合模板"""

from dataclasses import dataclass, field


@dataclass
class ChainStep:
    tool_intent: str
    depends_on: list[str] = field(default_factory=list)
    optional: bool = False


@dataclass
class ToolChain:
    name: str
    description: str
    steps: list[ChainStep]


STANDARD_TRIAL = ToolChain(
    name="standard_trial",
    description="标准单轮实验：改配置 -> 训练 -> 读指标",
    steps=[
        ChainStep(tool_intent="config_update"),
        ChainStep(tool_intent="launch_train", depends_on=["config_update"]),
        ChainStep(tool_intent="read_metrics", depends_on=["launch_train"]),
        ChainStep(tool_intent="read_logs", depends_on=["launch_train"], optional=True),
    ],
)

DIAGNOSTIC_RUN = ToolChain(
    name="diagnostic_run",
    description="诊断性运行：读日志 -> 读指标 -> 对比",
    steps=[
        ChainStep(tool_intent="read_logs"),
        ChainStep(tool_intent="read_metrics"),
        ChainStep(
            tool_intent="compare_runs",
            depends_on=["read_logs", "read_metrics"],
        ),
    ],
)

RECOVERY_TRIAL = ToolChain(
    name="recovery_trial",
    description="恢复性实验：回滚/降配 -> 重新训练 -> 读指标",
    steps=[
        ChainStep(tool_intent="config_update"),
        ChainStep(tool_intent="launch_train", depends_on=["config_update"]),
        ChainStep(tool_intent="read_metrics", depends_on=["launch_train"]),
    ],
)

REPO_ANALYSIS = ToolChain(
    name="repo_analysis",
    description="项目分析：扫描结构 -> 读配置 -> 读训练脚本",
    steps=[
        ChainStep(tool_intent="inspect_repo"),
        ChainStep(tool_intent="config_read", depends_on=["inspect_repo"]),
        ChainStep(tool_intent="read_file", depends_on=["inspect_repo"]),
    ],
)

ALL_CHAINS: dict[str, ToolChain] = {
    c.name: c
    for c in [STANDARD_TRIAL, DIAGNOSTIC_RUN, RECOVERY_TRIAL, REPO_ANALYSIS]
}