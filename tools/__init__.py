"""tools 包 — 提供 build_default_registry 工厂方法"""

from tools.impl.config_tools import PatchConfigTool, ReadConfigTool
from tools.impl.exec_tools import LaunchTrainTool
from tools.impl.file_tools import InspectRepoTool, ReadFileTool
from tools.impl.metric_tools import CompareRunsTool, ReadLogsTool, ReadMetricsTool
from tools.registry import ToolRegistry


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(InspectRepoTool())
    registry.register(ReadFileTool())
    registry.register(ReadConfigTool())
    registry.register(PatchConfigTool())
    registry.register(LaunchTrainTool())
    registry.register(ReadMetricsTool())
    registry.register(ReadLogsTool())
    registry.register(CompareRunsTool())
    return registry