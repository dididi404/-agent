"""Tool Guard — 安全策略层，在工具执行前进行安全检查"""

from schemas.common import RiskLevel
from schemas.task import ConstraintSpec
from tools.base import BaseTool


class ToolGuard:
    def __init__(self, constraints: ConstraintSpec | None = None):
        self.constraints = constraints or ConstraintSpec()
        self._auto_approve_levels = {RiskLevel.LOW}
        self._blocked_tools: set[str] = set()

    def check(self, tool: BaseTool, params: dict) -> tuple[bool, str]:
        if tool.meta.name in self._blocked_tools:
            return False, f"Tool '{tool.meta.name}' is blocked by policy"

        if not self.constraints.allow_config_edit and tool.meta.name == "patch_config":
            return False, "Config editing is not allowed by task constraints"

        if not self.constraints.allow_code_edit and tool.meta.name in (
            "edit_code",
            "write_file",
        ):
            return False, "Code editing is not allowed by task constraints"

        cmd = params.get("cmd", "")
        for blocked in self.constraints.blocked_commands:
            if blocked in cmd:
                return False, f"Command contains blocked pattern: '{blocked}'"

        if tool.meta.risk_level not in self._auto_approve_levels:
            if tool.meta.approval_required:
                return False, f"Tool '{tool.meta.name}' requires human approval (risk={tool.meta.risk_level.value})"

        return True, ""

    def block_tool(self, name: str) -> None:
        self._blocked_tools.add(name)

    def set_auto_approve(self, *levels: RiskLevel) -> None:
        self._auto_approve_levels = set(levels)