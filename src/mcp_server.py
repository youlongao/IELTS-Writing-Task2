"""MCP server wrapper for the IELTS Task 2 agent.

This module implements a lightweight JSON-RPC MCP server over stdio without
requiring the external MCP Python SDK. It exposes the project's main product
interfaces as MCP tools:

- Agent workflow: generate ideas, deepen an idea, evaluate an outline or essay.
- RAG: retrieve context, rebuild the reference store, inspect collection counts.
- Harness: evaluate generated sample essays and compare providers with a fixed judge.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from .agent.state import Scenario, WorkflowState
from .agent.workflow import AgentWorkflow
from .harness.compare import compare_results
from .harness.evaluator import Evaluator
from .harness.evaluator import _build_generation_pipeline, _build_pipeline
from .llm.factory import LLMFactory
from .prompts.templates import PromptManager
from .rag.embedder import Embedder
from .rag.loader import KnowledgeLoader
from .rag.retriever import Retriever
from .rag.store import VectorStore
from .utils.config import Config


JSONDict = Dict[str, Any]


class MCPError(Exception):
    """Error that should be returned to the MCP client."""

    def __init__(self, message: str, code: int = -32000):
        super().__init__(message)
        self.code = code


def _json_text(payload: Any) -> list[JSONDict]:
    return [
        {
            "type": "text",
            "text": json.dumps(payload, ensure_ascii=False, indent=2),
        }
    ]


def _schema(properties: JSONDict, required: Optional[list[str]] = None) -> JSONDict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _state_to_dict(state: WorkflowState) -> JSONDict:
    data = asdict(state)
    data["scenario"] = state.scenario.value if isinstance(state.scenario, Scenario) else state.scenario
    return data


def _safe_dataclass(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


class IeltsMCPServer:
    """Tool registry and handlers for the IELTS MCP server."""

    def __init__(self):
        self.config = Config()
        self._retriever: Optional[Retriever] = None
        self.tools: Dict[str, Callable[[JSONDict], JSONDict]] = {
            "ielts_generate": self.tool_generate,
            "ielts_deepen_idea": self.tool_deepen_idea,
            "ielts_evaluate_outline": self.tool_evaluate_outline,
            "ielts_evaluate_essay": self.tool_evaluate_essay,
            "rag_retrieve_context": self.tool_retrieve_context,
            "rag_build_knowledge_base": self.tool_build_knowledge_base,
            "rag_collection_counts": self.tool_collection_counts,
            "harness_rule_evaluate": self.tool_harness_rule_evaluate,
            "harness_evaluate_sample_essay": self.tool_harness_evaluate_sample_essay,
            "harness_compare_sample_essay": self.tool_harness_compare_sample_essay,
        }

    def list_tools(self) -> list[JSONDict]:
        return [
            {
                "name": "ielts_generate",
                "description": "Generate IELTS Task 2 question analysis, writable ideas, RAG context, and an outline.",
                "inputSchema": _schema(
                    {
                        "question": {"type": "string", "description": "IELTS Task 2 question."},
                        "provider": {"type": "string", "enum": ["openai", "deepseek"], "description": "LLM provider."},
                        "model": {"type": "string", "description": "Optional model override."},
                        "stance": {"type": "string", "description": "Optional selected writing path or position."},
                    },
                    ["question"],
                ),
            },
            {
                "name": "ielts_deepen_idea",
                "description": "Develop a shallow learner idea into an IELTS-style argument chain.",
                "inputSchema": _schema(
                    {
                        "idea": {"type": "string", "description": "Learner's idea."},
                        "question": {"type": "string", "description": "Original IELTS question."},
                        "provider": {"type": "string", "enum": ["openai", "deepseek"]},
                        "model": {"type": "string"},
                    },
                    ["idea"],
                ),
            },
            {
                "name": "ielts_evaluate_outline",
                "description": "Evaluate a learner outline as an idea-and-structure plan.",
                "inputSchema": _schema(
                    {
                        "question": {"type": "string"},
                        "outline": {"type": "string"},
                        "provider": {"type": "string", "enum": ["openai", "deepseek"]},
                        "model": {"type": "string"},
                    },
                    ["question", "outline"],
                ),
            },
            {
                "name": "ielts_evaluate_essay",
                "description": "Evaluate a learner essay for idea quality and actual IELTS band estimate.",
                "inputSchema": _schema(
                    {
                        "question": {"type": "string"},
                        "essay": {"type": "string"},
                        "outline": {"type": "string", "description": "Optional generated outline reference."},
                        "provider": {"type": "string", "enum": ["openai", "deepseek"]},
                        "model": {"type": "string"},
                    },
                    ["question", "essay"],
                ),
            },
            {
                "name": "rag_retrieve_context",
                "description": "Retrieve relevant writing reference chunks from the RAG store.",
                "inputSchema": _schema(
                    {
                        "query": {"type": "string"},
                        "topic": {"type": "string"},
                        "question_type": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                        "collections": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional collections, e.g. rubrics, topics, vocabulary.",
                        },
                        "max_chars": {"type": "integer", "minimum": 100, "maximum": 10000},
                    },
                    ["query"],
                ),
            },
            {
                "name": "rag_build_knowledge_base",
                "description": "Rebuild the Chroma vector store from data/knowledge_base.",
                "inputSchema": _schema({}),
            },
            {
                "name": "rag_collection_counts",
                "description": "Return document counts for every RAG collection.",
                "inputSchema": _schema({}),
            },
            {
                "name": "harness_rule_evaluate",
                "description": "Run deterministic rule checks on an agent output without calling an LLM judge.",
                "inputSchema": _schema(
                    {
                        "question": {"type": "string"},
                        "output": {"type": "string"},
                    },
                    ["question", "output"],
                ),
            },
            {
                "name": "harness_evaluate_sample_essay",
                "description": "Generate a sample essay for one IELTS Task 2 question, then evaluate it with a fixed judge.",
                "inputSchema": _schema(
                    {
                        "question": {"type": "string"},
                        "provider": {"type": "string", "enum": ["openai", "deepseek"], "description": "Generation provider."},
                        "judge_provider": {"type": "string", "enum": ["openai", "deepseek"], "description": "Fixed judge provider."},
                    },
                    ["question"],
                ),
            },
            {
                "name": "harness_compare_sample_essay",
                "description": "Compare two generation providers on the same IELTS question while keeping one fixed judge.",
                "inputSchema": _schema(
                    {
                        "question": {"type": "string"},
                        "provider_a": {"type": "string", "enum": ["openai", "deepseek"]},
                        "provider_b": {"type": "string", "enum": ["openai", "deepseek"]},
                        "judge_provider": {"type": "string", "enum": ["openai", "deepseek"]},
                    },
                    ["question"],
                ),
            },
        ]

    def handle(self, request: JSONDict) -> Optional[JSONDict]:
        method = request.get("method")
        request_id = request.get("id")

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "ielts-task2-agent",
                        "version": "0.1.0",
                    },
                }
            elif method == "notifications/initialized":
                return None
            elif method == "tools/list":
                result = {"tools": self.list_tools()}
            elif method == "tools/call":
                params = request.get("params") or {}
                result = self.call_tool(params.get("name", ""), params.get("arguments") or {})
            else:
                raise MCPError(f"Unsupported method: {method}", code=-32601)

            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except MCPError as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": exc.code, "message": str(exc)},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": str(exc)},
            }

    def call_tool(self, name: str, arguments: JSONDict) -> JSONDict:
        handler = self.tools.get(name)
        if not handler:
            raise MCPError(f"Unknown tool: {name}", code=-32602)
        payload = handler(arguments)
        return {"content": _json_text(payload)}

    def build_workflow(self, provider: Optional[str] = None, model: Optional[str] = None) -> AgentWorkflow:
        provider_name = provider or self.config.default_provider
        overrides = {"model_name": model} if model else None
        llm = LLMFactory.create(provider_name, self.config, overrides)
        return AgentWorkflow(
            llm=llm,
            retriever=self.get_retriever(),
            prompt_manager=PromptManager(),
        )

    def get_retriever(self) -> Retriever:
        if self._retriever is None:
            embedder = Embedder(self.config.get("rag.embedding_model", "all-MiniLM-L6-v2"))
            store = VectorStore(self.config.chroma_persist_dir)
            self._retriever = Retriever(embedder, store)
        return self._retriever

    def tool_generate(self, args: JSONDict) -> JSONDict:
        question = self._required(args, "question")
        workflow = self.build_workflow(args.get("provider"), args.get("model"))
        state = WorkflowState(
            user_input=question,
            scenario=Scenario.GENERATE,
            selected_stance=args.get("stance"),
        )
        return _state_to_dict(workflow.run(state))

    def tool_deepen_idea(self, args: JSONDict) -> JSONDict:
        idea = self._required(args, "idea")
        question = args.get("question") or idea
        workflow = self.build_workflow(args.get("provider"), args.get("model"))
        state = WorkflowState(
            user_input=question,
            scenario=Scenario.DEEPEN,
            user_idea=idea,
        )
        return _state_to_dict(workflow.run(state))

    def tool_evaluate_outline(self, args: JSONDict) -> JSONDict:
        question = self._required(args, "question")
        outline = self._required(args, "outline")
        workflow = self.build_workflow(args.get("provider"), args.get("model"))
        state = WorkflowState(
            user_input=question,
            scenario=Scenario.EVALUATE,
            user_outline=outline,
        )
        return _state_to_dict(workflow.run(state))

    def tool_evaluate_essay(self, args: JSONDict) -> JSONDict:
        question = self._required(args, "question")
        essay = self._required(args, "essay")
        workflow = self.build_workflow(args.get("provider"), args.get("model"))
        rag_context = args.get("rag_context", "")
        if not rag_context and workflow.retriever:
            results = workflow.retriever.retrieve(query=f"{question}\n\n{essay}")
            rag_context = workflow.retriever.format_context(results)
        return {
            "feedback": workflow.evaluate_essay(
                question=question,
                essay=essay,
                outline=args.get("outline", ""),
                rag_context=rag_context,
            )
        }

    def tool_retrieve_context(self, args: JSONDict) -> JSONDict:
        query = self._required(args, "query")
        top_k = int(args.get("top_k", 3))
        max_chars = int(args.get("max_chars", 2200))
        retriever = self.get_retriever()
        results = retriever.retrieve(
            query=query,
            topic=args.get("topic"),
            question_type=args.get("question_type"),
            top_k=top_k,
            collections=args.get("collections"),
        )
        return {
            "context": retriever.format_context(results, max_chars=max_chars),
            "results": results,
        }

    def tool_build_knowledge_base(self, args: JSONDict) -> JSONDict:
        retriever = self.get_retriever()
        loader = KnowledgeLoader(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        retriever.build_knowledge_base(loader)
        return self.tool_collection_counts({})

    def tool_collection_counts(self, args: JSONDict) -> JSONDict:
        store = self.get_retriever().store
        counts = {}
        for collection_name in store.COLLECTIONS:
            try:
                counts[collection_name] = store.count(collection_name)
            except Exception as exc:
                counts[collection_name] = {"error": str(exc)}
        return {"collections": counts}

    def tool_harness_rule_evaluate(self, args: JSONDict) -> JSONDict:
        question = self._required(args, "question")
        output = self._required(args, "output")
        return Evaluator.rule_evaluate(question, output)

    def tool_harness_evaluate_sample_essay(self, args: JSONDict) -> JSONDict:
        question = self._required(args, "question")
        provider = args.get("provider") or self.config.default_provider
        judge_provider = args.get("judge_provider") or self.config.get("harness.judge_model", "openai")
        judge, pipeline = _build_pipeline(provider, judge_provider)
        evaluator = Evaluator(judge)
        cases = [{
            "id": "mcp_manual_001",
            "question": question,
            "source": "mcp_manual",
        }]
        results = evaluator.batch_evaluate(cases, pipeline)
        return {
            "harness_target": "sample_essay",
            "provider": provider,
            "judge_provider": judge_provider,
            "results": results,
        }

    def tool_harness_compare_sample_essay(self, args: JSONDict) -> JSONDict:
        question = self._required(args, "question")
        provider_a = args.get("provider_a") or "deepseek"
        provider_b = args.get("provider_b") or "openai"
        judge_provider = args.get("judge_provider") or "openai"

        judge = LLMFactory.create(judge_provider, self.config)
        evaluator = Evaluator(judge)
        cases = [{
            "id": "mcp_manual_001",
            "question": question,
            "source": "mcp_manual",
        }]
        pipeline_a = _build_generation_pipeline(provider_a)
        pipeline_b = _build_generation_pipeline(provider_b)
        results_a = evaluator.batch_evaluate(cases, pipeline_a)
        results_b = evaluator.batch_evaluate(cases, pipeline_b)
        comparison = compare_results(provider_a, results_a, provider_b, results_b, judge_provider)
        return {
            "harness_target": "sample_essay",
            "comparison": comparison,
            "provider_a_results": results_a,
            "provider_b_results": results_b,
        }

    @staticmethod
    def _required(args: JSONDict, key: str) -> str:
        value = args.get(key)
        if not isinstance(value, str) or not value.strip():
            raise MCPError(f"Missing required string argument: {key}", code=-32602)
        return value.strip()


def _read_message(stream) -> Optional[JSONDict]:
    """Read one MCP message using Content-Length framing."""
    headers: Dict[str, str] = {}

    while True:
        line = stream.buffer.readline()
        if not line:
            return None
        line_text = line.decode("utf-8").strip()
        if not line_text:
            break
        key, _, value = line_text.partition(":")
        headers[key.lower()] = value.strip()

    length = int(headers.get("content-length", "0"))
    if length <= 0:
        return None

    body = stream.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def _write_message(stream, payload: JSONDict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    stream.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    stream.buffer.write(body)
    stream.buffer.flush()


def serve_stdio(server: Optional[IeltsMCPServer] = None) -> None:
    server = server or IeltsMCPServer()
    while True:
        request = _read_message(sys.stdin)
        if request is None:
            break
        response = server.handle(request)
        if response is not None:
            _write_message(sys.stdout, response)


def main(argv: Optional[Iterable[str]] = None) -> int:
    _ = argv
    serve_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
