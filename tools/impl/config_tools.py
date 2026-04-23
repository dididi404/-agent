"""config_tools — read_config / patch_config 工具实现"""

import copy
import shutil
from pathlib import Path
from typing import Any

import yaml

from schemas.common import RiskLevel
from tools.base import BaseTool, PostValidation, ToolMeta
from tools.schemas import (
    PatchConfigInput,
    PatchConfigOutput,
    ReadConfigInput,
    ReadConfigOutput,
)


def _get_nested(data: dict, dotted_key: str) -> Any:
    keys = dotted_key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict) and k in current:
            current = current[k]
        else:
            return None
    return current


def _set_nested(data: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    current = data
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


class ReadConfigTool(BaseTool):
    meta = ToolMeta(
        name="read_config",
        description="读取并解析 YAML/JSON 配置文件",
        input_schema=ReadConfigInput,
        output_schema=ReadConfigOutput,
        risk_level=RiskLevel.LOW,
        timeout_sec=10,
        is_idempotent=True,
        preconditions=["config_path must exist"],
        side_effects=[],
    )

    def check_preconditions(self, params: ReadConfigInput) -> tuple[bool, str]:
        if not Path(params.config_path).is_file():
            return False, f"Config file not found: {params.config_path}"
        return True, ""

    def execute(self, params: ReadConfigInput) -> ReadConfigOutput:
        p = Path(params.config_path)
        content = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(content)
            return ReadConfigOutput(data=data, format="yaml")
        elif p.suffix == ".json":
            import json
            data = json.loads(content)
            return ReadConfigOutput(data=data, format="json")
        raise ValueError(f"Unsupported config format: {p.suffix}")


class PatchConfigTool(BaseTool):
    meta = ToolMeta(
        name="patch_config",
        description="修改 YAML 配置文件中的指定字段（自动备份原文件）",
        input_schema=PatchConfigInput,
        output_schema=PatchConfigOutput,
        risk_level=RiskLevel.MEDIUM,
        timeout_sec=10,
        is_idempotent=False,
        preconditions=["config_path must exist"],
        rollback_hint="restore from .bak file",
        failure_types=["file_not_found", "parse_error", "write_error"],
        side_effects=["modifies config file on disk"],
        post_validations=[
            PostValidation(
                check="re-read config and verify patch was applied",
                on_fail="rollback_to_backup",
            ),
        ],
        agent_visibility=["executor_agent"],
    )

    def check_preconditions(self, params: PatchConfigInput) -> tuple[bool, str]:
        if not Path(params.config_path).is_file():
            return False, f"Config file not found: {params.config_path}"
        return True, ""

    def execute(self, params: PatchConfigInput) -> PatchConfigOutput:
        p = Path(params.config_path)
        content = p.read_text(encoding="utf-8")
        data = yaml.safe_load(content)

        backup_path = None
        if params.backup:
            backup_path = str(p) + ".bak"
            shutil.copy2(p, backup_path)

        original_values = {}
        new_values = {}

        for dotted_key, value in params.patch.items():
            old_val = _get_nested(data, dotted_key)
            original_values[dotted_key] = old_val
            _set_nested(data, dotted_key, value)
            new_values[dotted_key] = value

        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        return PatchConfigOutput(
            original_values=original_values,
            new_values=new_values,
            backup_path=backup_path,
        )

    def post_validate(self, result: PatchConfigOutput) -> tuple[bool, str]:
        if not result.new_values:
            return False, "No values were patched"
        return True, ""