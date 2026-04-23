"""file_tools — inspect_repo / read_file 工具实现"""

import os
from pathlib import Path

from tools.base import BaseTool, PostValidation, ToolMeta
from tools.schemas import (
    InspectRepoInput,
    InspectRepoOutput,
    ReadFileInput,
    ReadFileOutput,
)
from schemas.common import RiskLevel


class InspectRepoTool(BaseTool):
    meta = ToolMeta(
        name="inspect_repo",
        description="扫描项目目录结构，识别配置文件和训练脚本",
        input_schema=InspectRepoInput,
        output_schema=InspectRepoOutput,
        risk_level=RiskLevel.LOW,
        timeout_sec=30,
        is_idempotent=True,
        preconditions=["repo_path must exist and be a directory"],
        side_effects=[],
        agent_visibility=["repo_agent", "executor_agent"],
    )

    def check_preconditions(self, params: InspectRepoInput) -> tuple[bool, str]:
        if not Path(params.repo_path).is_dir():
            return False, f"Directory not found: {params.repo_path}"
        return True, ""

    def execute(self, params: InspectRepoInput) -> InspectRepoOutput:
        repo = Path(params.repo_path)
        files = []
        directories = []
        config_candidates = []
        script_candidates = []

        for item in sorted(repo.rglob("*")):
            if any(p.startswith(".") for p in item.parts):
                continue
            if "__pycache__" in str(item) or "node_modules" in str(item):
                continue

            rel = str(item.relative_to(repo))
            if item.is_dir():
                directories.append(rel)
            else:
                files.append(rel)
                suffix = item.suffix.lower()
                name = item.name.lower()
                if suffix in (".yaml", ".yml", ".json", ".toml", ".cfg", ".ini"):
                    config_candidates.append(rel)
                if suffix == ".py" and ("train" in name or "main" in name or "run" in name):
                    script_candidates.append(rel)

        return InspectRepoOutput(
            files=files,
            directories=directories,
            has_requirements=any("requirements" in f for f in files),
            has_setup_py="setup.py" in files,
            has_pyproject="pyproject.toml" in files,
            config_candidates=config_candidates,
            script_candidates=script_candidates,
        )


class ReadFileTool(BaseTool):
    meta = ToolMeta(
        name="read_file",
        description="读取指定文件内容，支持截断",
        input_schema=ReadFileInput,
        output_schema=ReadFileOutput,
        risk_level=RiskLevel.LOW,
        timeout_sec=10,
        is_idempotent=True,
        preconditions=["file_path must exist"],
        side_effects=[],
    )

    def check_preconditions(self, params: ReadFileInput) -> tuple[bool, str]:
        if not Path(params.file_path).is_file():
            return False, f"File not found: {params.file_path}"
        return True, ""

    def execute(self, params: ReadFileInput) -> ReadFileOutput:
        p = Path(params.file_path)
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)
        truncated = total > params.max_lines
        if truncated:
            lines = lines[: params.max_lines]
        return ReadFileOutput(
            content="\n".join(lines),
            line_count=total,
            truncated=truncated,
        )