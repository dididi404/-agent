"""工具注册表 — 注册所有工具并提供查询接口"""

from tools.base import BaseTool, ToolMeta


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.meta.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def get_or_raise(self, name: str) -> BaseTool:
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools.keys()))
            raise KeyError(f"Tool '{name}' not found. Available: {available}")
        return tool

    def list_tools(self) -> list[ToolMeta]:
        return [t.meta for t in self._tools.values()]

    def list_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def list_for_agent(self, agent_name: str) -> list[ToolMeta]:
        result = []
        for tool in self._tools.values():
            if "*" in tool.meta.agent_visibility or agent_name in tool.meta.agent_visibility:
                result.append(tool.meta)
        return result

    @property
    def size(self) -> int:
        return len(self._tools)