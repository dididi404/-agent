"""workflow/graph.py — LangGraph 图定义

将状态机节点组装成可执行的 StateGraph。
状态流转：
  init -> analyze_repo -> plan -> execute -> assess -> decide
                                    ^                    |
                                    |  (continue)        |
                                    +--------------------+
                                    |  (next_trial/recover) -> plan
                                    +-- (finish) -> finish -> END
"""

from langgraph.graph import END, START, StateGraph

from tools.dispatcher import ToolDispatcher
from tools.guard import ToolGuard
from tools.registry import ToolRegistry
from workflow.state import GraphState
from workflow.supervisor import SupervisorNodes, route_after_decide


def build_graph(
    dispatcher: ToolDispatcher,
    repo_agent=None,
    planner_agent=None,
    use_llm: bool = False,
) -> StateGraph:
    nodes = SupervisorNodes(
        dispatcher,
        repo_agent=repo_agent,
        planner_agent=planner_agent,
        use_llm=use_llm,
    )

    graph = StateGraph(GraphState)

    graph.add_node("init", nodes.init_node)
    graph.add_node("analyze_repo", nodes.analyze_repo_node)
    graph.add_node("plan", nodes.plan_node)
    graph.add_node("execute", nodes.execute_node)
    graph.add_node("assess", nodes.assess_node)
    graph.add_node("decide", nodes.decide_node)
    graph.add_node("finish", nodes.finish_node)

    graph.add_edge(START, "init")
    graph.add_edge("init", "analyze_repo")
    graph.add_edge("analyze_repo", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "assess")
    graph.add_edge("assess", "decide")
    graph.add_conditional_edges("decide", route_after_decide, {
        "execute": "execute",
        "plan": "plan",
        "finish": "finish",
    })
    graph.add_edge("finish", END)

    return graph


def create_runnable(
    registry: ToolRegistry,
    guard: ToolGuard,
    repo_agent=None,
    planner_agent=None,
    use_llm: bool = False,
):
    dispatcher = ToolDispatcher(registry, guard)
    graph = build_graph(
        dispatcher,
        repo_agent=repo_agent,
        planner_agent=planner_agent,
        use_llm=use_llm,
    )
    return graph.compile(), dispatcher
