"""workflow/supervisor.py — 各状态机节点的具体逻辑

支持两种模式：
  - use_llm=True:  用 LLM Agent 做 repo 分析和实验规划
  - use_llm=False: 用硬编码规则（Step 3 骨架模式，也作为 LLM 失败的降级兜底）
"""

import uuid
from datetime import datetime
from typing import Any

from schemas.common import TaskPhase, ToolStatus, TrialStatus
from schemas.plan import ExperimentPlan, PlanStep
from schemas.repo import RepoProfile, SearchSpaceCandidate
from schemas.state import BestResult, RunState, TrialRecord
from schemas.trial import TrialAssessment
from tools.dispatcher import ToolDispatcher
from workflow.state import GraphState


class SupervisorNodes:
    """提供所有状态机节点函数，共享 dispatcher 等基础设施。"""

    def __init__(
        self,
        dispatcher: ToolDispatcher,
        repo_agent=None,
        planner_agent=None,
        use_llm: bool = False,
    ):
        self.dispatcher = dispatcher
        self.repo_agent = repo_agent
        self.planner_agent = planner_agent
        self.use_llm = use_llm and repo_agent is not None and planner_agent is not None

    # ------------------------------------------------------------------ init
    def init_node(self, state: GraphState) -> dict:
        task = state["task_spec"]
        run_state = RunState(task_id=task.task_id, phase=TaskPhase.INIT)
        return {
            "run_state": run_state,
            "repo_profile": None,
            "current_plan": None,
            "current_step_index": 0,
            "current_tool_result": None,
            "current_assessment": None,
            "error": None,
            "decision": "",
            "messages": [{"role": "system", "content": f"Task {task.task_id} initialized (llm={'on' if self.use_llm else 'off'})"}],
        }

    # --------------------------------------------------------- analyze_repo
    def analyze_repo_node(self, state: GraphState) -> dict:
        run_state = state["run_state"].model_copy()
        run_state.phase = TaskPhase.ANALYZING_REPO
        task = state["task_spec"]

        if self.use_llm:
            return self._analyze_repo_llm(state, run_state, task)
        return self._analyze_repo_hardcoded(state, run_state, task)

    def _analyze_repo_llm(self, state, run_state, task) -> dict:
        try:
            profile, meta = self.repo_agent.analyze(task.repo_path)
            run_state.total_tokens_used += meta.get("tokens_used", 0)
            return {
                "run_state": run_state,
                "repo_profile": profile,
                "error": None,
                "messages": state.get("messages", []) + [
                    {"role": "repo_agent", "content": f"[LLM] Found {len(profile.search_space_candidates)} params, "
                     f"{len(profile.config_files)} configs (tokens={meta.get('tokens_used', 0)})"}
                ],
            }
        except Exception as e:
            return {
                "run_state": run_state,
                "error": None,
                "messages": state.get("messages", []) + [
                    {"role": "repo_agent", "content": f"[LLM] Failed ({e}), falling back to hardcoded"}
                ],
                **self._analyze_repo_hardcoded(state, run_state, task),
            }

    def _analyze_repo_hardcoded(self, state, run_state, task) -> dict:
        inspect_result = self.dispatcher.dispatch(
            intent="inspect_repo",
            params={"repo_path": task.repo_path},
            caller_agent="repo_agent",
        )
        if inspect_result.status != ToolStatus.SUCCESS:
            return {
                "run_state": run_state,
                "error": f"inspect_repo failed: {inspect_result.error_message}",
                "error_category": "tool_error",
            }

        config_files = []
        config_path = None
        for candidate in ["configs/base.yaml", "config.yaml", "configs/config.yaml"]:
            cfg_result = self.dispatcher.dispatch(
                intent="config_read",
                params={"config_path": f"{task.repo_path}/{candidate}"},
                caller_agent="repo_agent",
            )
            if cfg_result.status == ToolStatus.SUCCESS:
                config_path = f"{task.repo_path}/{candidate}"
                config_files.append(candidate)
                break

        metric_sources = []
        import os
        for candidate in ["outputs/metrics.json", "metrics.json", "outputs/results.json"]:
            if os.path.exists(f"{task.repo_path}/{candidate}"):
                metric_sources.append(candidate)

        search_space = []
        if config_path:
            from tools.impl.config_tools import ReadConfigTool
            from tools.schemas import ReadConfigInput
            tool = ReadConfigTool()
            cfg_data = tool.execute(ReadConfigInput(config_path=config_path))
            opt_cfg = cfg_data.data.get("optimizer", {})
            train_cfg = cfg_data.data.get("training", {})
            sched_cfg = cfg_data.data.get("scheduler", {})
            if "lr" in opt_cfg:
                search_space.append(SearchSpaceCandidate(
                    name="lr", type="float",
                    current_value=opt_cfg["lr"],
                    config_path=config_path, config_key="optimizer.lr",
                ))
            if "weight_decay" in opt_cfg:
                search_space.append(SearchSpaceCandidate(
                    name="weight_decay", type="float",
                    current_value=opt_cfg["weight_decay"],
                    config_path=config_path, config_key="optimizer.weight_decay",
                ))
            if "batch_size" in train_cfg:
                search_space.append(SearchSpaceCandidate(
                    name="batch_size", type="int",
                    current_value=train_cfg["batch_size"],
                    config_path=config_path, config_key="training.batch_size",
                ))
            if not sched_cfg.get("enabled", True):
                search_space.append(SearchSpaceCandidate(
                    name="scheduler_enabled", type="bool",
                    current_value=False,
                    config_path=config_path, config_key="scheduler.enabled",
                ))

        profile = RepoProfile(
            train_entry=f"python train.py --config {config_files[0]}" if config_files else "python train.py",
            config_files=config_files,
            metric_sources=metric_sources,
            framework="pytorch",
            search_space_candidates=search_space,
        )

        return {
            "run_state": run_state,
            "repo_profile": profile,
            "error": None,
            "messages": state.get("messages", []) + [
                {"role": "repo_agent", "content": f"[hardcoded] Found {len(search_space)} tunable params, {len(config_files)} config files"}
            ],
        }

    # -------------------------------------------------------------- plan
    def plan_node(self, state: GraphState) -> dict:
        run_state = state["run_state"].model_copy()
        run_state.phase = TaskPhase.PLANNING
        profile = state.get("repo_profile")
        task = state["task_spec"]

        if not profile or not profile.search_space_candidates:
            return {
                "run_state": run_state,
                "error": "No tunable parameters found in repo",
                "error_category": "experiment_error",
            }

        if self.use_llm:
            return self._plan_llm(state, run_state, task, profile)
        return self._plan_hardcoded(state, run_state, task, profile)

    def _plan_llm(self, state, run_state, task, profile) -> dict:
        try:
            plan, meta = self.planner_agent.plan(
                task=task,
                profile=profile,
                run_state=run_state,
                memory_context=None,
            )
            run_state.active_plan_id = plan.plan_id
            run_state.total_tokens_used += meta.get("tokens_used", 0)
            return {
                "run_state": run_state,
                "current_plan": plan,
                "current_step_index": 0,
                "error": None,
                "messages": state.get("messages", []) + [
                    {"role": "planner_agent", "content": f"[LLM] Plan {plan.plan_id}: {plan.hypothesis} (tokens={meta.get('tokens_used', 0)})"}
                ],
            }
        except Exception as e:
            return {
                "run_state": run_state,
                "error": None,
                "messages": state.get("messages", []) + [
                    {"role": "planner_agent", "content": f"[LLM] Failed ({e}), falling back to hardcoded"}
                ],
                **self._plan_hardcoded(state, run_state, task, profile),
            }

    def _plan_hardcoded(self, state, run_state, task, profile) -> dict:
        plan_id = f"plan_{uuid.uuid4().hex[:6]}"
        config_file = f"{task.repo_path}/{profile.config_files[0]}" if profile.config_files else ""

        patch = {}
        hypothesis_parts = []
        for param in profile.search_space_candidates:
            if param.name == "lr" and param.current_value is not None:
                new_lr = float(param.current_value) * 30
                patch["optimizer.lr"] = new_lr
                hypothesis_parts.append(f"increase lr from {param.current_value} to {new_lr}")
            elif param.name == "batch_size" and param.current_value is not None:
                new_bs = min(int(param.current_value) * 4, 256)
                patch["training.batch_size"] = new_bs
                hypothesis_parts.append(f"increase batch_size from {param.current_value} to {new_bs}")
            elif param.name == "scheduler_enabled" and param.current_value is False:
                patch["scheduler.enabled"] = True
                patch["scheduler.warmup_steps"] = 100
                hypothesis_parts.append("enable cosine scheduler with warmup")

        hypothesis = "Baseline is undertrained. " + "; ".join(hypothesis_parts) if hypothesis_parts else "Run baseline experiment"

        steps = []
        if patch:
            steps.append(PlanStep(
                step_id="s1", action="patch_config",
                tool_intent="config_update",
                params={"config_path": config_file, "patch": patch, "backup": True},
            ))
        steps.append(PlanStep(
            step_id="s2", action="run_training",
            tool_intent="launch_train",
            params={
                "cmd": f"cd {task.repo_path} && {profile.train_entry}",
                "working_dir": task.repo_path,
                "timeout_minutes": 30,
            },
            depends_on=["s1"] if patch else [],
        ))
        steps.append(PlanStep(
            step_id="s3", action="evaluate_result",
            tool_intent="read_metrics",
            params={"metrics_path": f"{task.repo_path}/outputs/metrics.json"},
            depends_on=["s2"],
        ))

        plan = ExperimentPlan(
            plan_id=plan_id,
            hypothesis=hypothesis,
            steps=steps,
            source_chain="standard_trial",
        )
        run_state.active_plan_id = plan_id

        return {
            "run_state": run_state,
            "current_plan": plan,
            "current_step_index": 0,
            "error": None,
            "messages": state.get("messages", []) + [
                {"role": "planner_agent", "content": f"[hardcoded] Plan {plan_id}: {hypothesis}"}
            ],
        }

    # ------------------------------------------------------------ execute
    def execute_node(self, state: GraphState) -> dict:
        run_state = state["run_state"].model_copy()
        run_state.phase = TaskPhase.EXECUTING_TRIAL
        plan = state.get("current_plan")
        step_idx = state.get("current_step_index", 0)

        if not plan or step_idx >= len(plan.steps):
            return {
                "run_state": run_state,
                "error": "No more steps to execute",
                "current_tool_result": None,
            }

        step = plan.steps[step_idx]
        result = self.dispatcher.dispatch(
            intent=step.tool_intent,
            params=step.params,
            caller_agent="executor_agent",
            intent_description=step.action,
        )

        if result.status in (ToolStatus.FAILED, ToolStatus.TIMEOUT) and not step.optional:
            run_state.last_error = result.error_message
            run_state.last_error_category = "tool_error"
            return {
                "run_state": run_state,
                "current_tool_result": result,
                "error": result.error_message,
                "error_category": "tool_error",
                "messages": state.get("messages", []) + [
                    {"role": "executor_agent", "content": f"Step {step.step_id} FAILED: {result.error_message}"}
                ],
            }

        next_idx = step_idx + 1
        return {
            "run_state": run_state,
            "current_tool_result": result,
            "current_step_index": next_idx,
            "error": None,
            "messages": state.get("messages", []) + [
                {"role": "executor_agent", "content": f"Step {step.step_id} ({step.tool_intent}): {result.status.value}"}
            ],
        }

    # -------------------------------------------------------------- assess
    def assess_node(self, state: GraphState) -> dict:
        run_state = state["run_state"].model_copy()
        run_state.phase = TaskPhase.ASSESSING_RESULT
        task = state["task_spec"]
        plan = state.get("current_plan")

        metric_result = self.dispatcher.dispatch(
            intent="read_metrics",
            params={"metrics_path": f"{task.repo_path}/outputs/metrics.json"},
            caller_agent="critic_agent",
        )

        if metric_result.status != ToolStatus.SUCCESS:
            assessment = TrialAssessment(
                trial_id=plan.plan_id if plan else "unknown",
                is_valid=False,
                confidence=0.0,
                diagnosis=f"Could not read metrics: {metric_result.error_message}",
            )
            return {"run_state": run_state, "current_assessment": assessment}

        from tools.impl.metric_tools import ReadMetricsTool
        from tools.schemas import ReadMetricsInput
        metrics_tool = ReadMetricsTool()
        metrics_out = metrics_tool.execute(
            ReadMetricsInput(metrics_path=f"{task.repo_path}/outputs/metrics.json")
        )

        summary = metrics_out.summary
        primary = task.success_criteria.primary_metric
        metric_value = summary.get(f"best_{primary}", summary.get(primary))

        if metric_value is None:
            for key, val in summary.items():
                if primary.replace("_", "") in key.replace("_", "") and isinstance(val, (int, float)):
                    metric_value = val
                    break

        trial_id = plan.plan_id if plan else f"trial_{uuid.uuid4().hex[:6]}"
        is_better = False
        if metric_value is not None and run_state.current_best is not None:
            is_better = task.is_better(metric_value, run_state.current_best.metric_value)
        elif metric_value is not None and run_state.current_best is None:
            is_better = True

        assessment = TrialAssessment(
            trial_id=trial_id,
            is_valid=metric_value is not None,
            metric_value=metric_value,
            metric_name=primary,
            better_than_best=is_better,
            confidence=0.8 if metric_value is not None else 0.0,
            diagnosis=f"{primary}={metric_value}" + (" (new best)" if is_better else ""),
        )

        trial_record = TrialRecord(
            trial_id=trial_id,
            plan_id=plan.plan_id if plan else "",
            status=TrialStatus.SUCCESS if metric_value is not None else TrialStatus.FAILED,
            metric_value=metric_value,
            metric_name=primary,
            started_at=datetime.now(),
            finished_at=datetime.now(),
        )
        run_state.attempted_trials.append(trial_record)

        if is_better and metric_value is not None:
            run_state.current_best = BestResult(
                trial_id=trial_id,
                metric_value=metric_value,
                metric_name=primary,
            )

        return {
            "run_state": run_state,
            "current_assessment": assessment,
            "messages": state.get("messages", []) + [
                {"role": "critic_agent", "content": f"Assessment: {assessment.diagnosis}, confidence={assessment.confidence}"}
            ],
        }

    # ----------------------------------------------------------- decide
    def decide_node(self, state: GraphState) -> dict:
        run_state = state["run_state"].model_copy()
        run_state.phase = TaskPhase.DECIDING_NEXT
        task = state["task_spec"]
        plan = state.get("current_plan")
        step_idx = state.get("current_step_index", 0)
        error = state.get("error")

        if error:
            if run_state.retry_count < 2:
                run_state.retry_count += 1
                return {"run_state": run_state, "decision": "recover", "error": None}
            else:
                run_state.phase = TaskPhase.FAILED
                return {"run_state": run_state, "decision": "finish"}

        if plan and step_idx < len(plan.steps):
            return {"run_state": run_state, "decision": "continue"}

        if run_state.trials_completed >= task.budget.max_trials:
            run_state.phase = TaskPhase.FINISHED
            return {"run_state": run_state, "decision": "finish"}

        assessment = state.get("current_assessment")
        if assessment and assessment.better_than_best:
            run_state.phase = TaskPhase.FINISHED
            return {
                "run_state": run_state,
                "decision": "finish",
                "messages": state.get("messages", []) + [
                    {"role": "supervisor", "content": f"Goal achieved! Best: {run_state.current_best}"}
                ],
            }

        run_state.retry_count = 0
        return {"run_state": run_state, "decision": "next_trial"}

    # ----------------------------------------------------------- finish
    def finish_node(self, state: GraphState) -> dict:
        run_state = state["run_state"].model_copy()
        if run_state.phase not in (TaskPhase.FINISHED, TaskPhase.FAILED):
            run_state.phase = TaskPhase.FINISHED
        return {
            "run_state": run_state,
            "messages": state.get("messages", []) + [
                {"role": "supervisor", "content": f"Task {run_state.task_id} completed. Phase: {run_state.phase.value}. "
                 f"Trials: {run_state.trials_completed}, Best: {run_state.current_best}. "
                 f"Tokens used: {run_state.total_tokens_used}"}
            ],
        }


def route_after_decide(state: GraphState) -> str:
    decision = state.get("decision", "finish")
    if decision == "continue":
        return "execute"
    elif decision == "next_trial":
        return "plan"
    elif decision == "recover":
        return "plan"
    else:
        return "finish"
