"""agents/repo_agent.py — 项目理解 Agent

用 LLM 分析 repo 结构并输出结构化 RepoProfile。
先用工具收集原始信息，再交给 LLM 做结构化理解。
"""

from schemas.common import ToolStatus
from schemas.repo import RepoProfile
from agents.base import BaseAgent
from llm.adapter import LLMAdapter
from llm.token_budget import TokenBudget
from tools.dispatcher import ToolDispatcher


class RepoAgent(BaseAgent):
    name = "repo_agent"
    output_schema = RepoProfile

    def __init__(
        self,
        llm: LLMAdapter,
        dispatcher: ToolDispatcher,
        token_budget: TokenBudget | None = None,
    ):
        super().__init__(llm, token_budget)
        self.dispatcher = dispatcher

    def analyze(self, repo_path: str) -> tuple[RepoProfile, dict]:
        inspect_result = self.dispatcher.dispatch(
            intent="inspect_repo",
            params={"repo_path": repo_path},
            caller_agent=self.name,
        )
        if inspect_result.status != ToolStatus.SUCCESS:
            raise RuntimeError(f"inspect_repo failed: {inspect_result.error_message}")

        from tools.impl.file_tools import InspectRepoTool
        from tools.schemas import InspectRepoInput
        inspector = InspectRepoTool()
        inspect_out = inspector.execute(InspectRepoInput(repo_path=repo_path))

        config_contents = {}
        for cfg_file in inspect_out.config_candidates:
            full_path = f"{repo_path}/{cfg_file}"
            r = self.dispatcher.dispatch(
                intent="config_read",
                params={"config_path": full_path},
                caller_agent=self.name,
            )
            if r.status == ToolStatus.SUCCESS:
                from tools.impl.config_tools import ReadConfigTool
                from tools.schemas import ReadConfigInput
                tool = ReadConfigTool()
                out = tool.execute(ReadConfigInput(config_path=full_path))
                config_contents[cfg_file] = out.data

        script_contents = {}
        for script in inspect_out.script_candidates[:3]:
            full_path = f"{repo_path}/{script}"
            r = self.dispatcher.dispatch(
                intent="read_file",
                params={"file_path": full_path, "max_lines": 200},
                caller_agent=self.name,
            )
            if r.status == ToolStatus.SUCCESS:
                from tools.impl.file_tools import ReadFileTool
                from tools.schemas import ReadFileInput
                tool = ReadFileTool()
                out = tool.execute(ReadFileInput(file_path=full_path, max_lines=200))
                script_contents[script] = out.content

        system_prompt = self.load_prompt("repo_agent.md")
        user_prompt = self.format_context(
            repo_path=repo_path,
            files=inspect_out.files,
            directories=inspect_out.directories,
            config_candidates=inspect_out.config_candidates,
            script_candidates=inspect_out.script_candidates,
            config_contents=config_contents,
            script_contents=script_contents,
        )

        profile, call_result = self.call_llm_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=RepoProfile,
        )

        return profile, {
            "tokens_used": call_result.total_tokens,
            "attempts": call_result.attempts,
            "files_inspected": len(inspect_out.files),
            "configs_read": len(config_contents),
            "scripts_read": len(script_contents),
        }
