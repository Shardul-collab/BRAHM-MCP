import traceback

import repositories.execution_repo as execution_repo

from tools.generate_queries import generate_queries
from tools.search_papers import search_papers
from tools.resolve_pdf import resolve_pdf
from tools.extract_lightweight_knowledge import extract_lightweight_knowledge  # S2_75
from tools.download_papers import download_papers
from tools.extract_paper_content import extract_paper_content
from tools.extract_research_knowledge import extract_research_knowledge
from tools.reconstruct_findings import reconstruct_findings


class ToolExecutor:

    def __init__(self, repo):
        self.repo = repo

        self.tools = {
            "generate_queries":               generate_queries,
            "search_papers":                  search_papers,
            "extract_lightweight_knowledge":  extract_lightweight_knowledge,  # S2_75
            "resolve_pdf":                    resolve_pdf,                    # S2_5
            "download_papers":                download_papers,                # S3
            "extract_paper_content":          extract_paper_content,          # S4
            "extract_research_knowledge":     extract_research_knowledge,     # S5
            "reconstruct_findings":           reconstruct_findings,           # S5.5
        }

    def execute(self, tool_name, workflow_id, kwargs=None):

        if kwargs is None:
            kwargs = {}

        if tool_name not in self.tools:
            raise ValueError(f"Tool not registered: {tool_name}")

        tool = self.tools[tool_name]

        try:
            return tool(self.repo, workflow_id, **kwargs)

        except Exception as e:

            error_trace = traceback.format_exc()

            print(f"\n❌ TOOL FAILURE: {tool_name}")
            print(error_trace)

            return {
                "status":    "error",
                "data":      None,
                "error":     str(e),
                "traceback": error_trace
            }
