"""Tests for the lightweight MCP server wrapper."""

import json
from unittest.mock import Mock, patch

from src.mcp_server import IeltsMCPServer


class TestIeltsMCPServer:
    """Test MCP JSON-RPC handlers."""

    def test_initialize(self):
        server = IeltsMCPServer()
        response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})

        assert response["id"] == 1
        assert response["result"]["serverInfo"]["name"] == "ielts-task2-agent"
        assert "tools" in response["result"]["capabilities"]

    def test_tools_list_contains_project_interfaces(self):
        server = IeltsMCPServer()
        response = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})

        names = {tool["name"] for tool in response["result"]["tools"]}
        assert "ielts_generate" in names
        assert "ielts_deepen_idea" in names
        assert "ielts_evaluate_outline" in names
        assert "ielts_evaluate_essay" in names
        assert "rag_retrieve_context" in names
        assert "rag_build_knowledge_base" in names
        assert "rag_collection_counts" in names
        assert "harness_rule_evaluate" in names
        assert "harness_evaluate_sample_essay" in names
        assert "harness_compare_sample_essay" in names

    def test_harness_rule_evaluate_tool(self):
        server = IeltsMCPServer()
        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "harness_rule_evaluate",
                    "arguments": {
                        "question": "Should governments invest in public transport?",
                        "output": (
                            "Question Type: Opinion\n"
                            "Position: agree\n"
                            "Main idea 1: reduce pollution. For example, buses reduce car use.\n"
                            "Main idea 2: reduce congestion.\n"
                            "Suggested Outline: introduction, body, conclusion."
                        ),
                    },
                },
            }
        )

        content = response["result"]["content"][0]
        assert content["type"] == "text"
        assert '"checks"' in content["text"]
        assert '"score"' in content["text"]

    def test_harness_evaluate_sample_essay_tool_uses_harness(self):
        server = IeltsMCPServer()
        fake_results = [{
            "test_case_id": "mcp_manual_001",
            "harness_target": "sample_essay",
            "scores": {"overall": 7.0},
            "error": None,
        }]

        with patch("src.mcp_server._build_pipeline", return_value=(Mock(model_name="judge"), lambda q: "essay")), \
             patch("src.mcp_server.Evaluator.batch_evaluate", return_value=fake_results) as mock_batch:
            response = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {
                        "name": "harness_evaluate_sample_essay",
                        "arguments": {
                            "question": "Should governments invest in public transport?",
                            "provider": "deepseek",
                            "judge_provider": "openai",
                        },
                    },
                }
            )

        payload = json.loads(response["result"]["content"][0]["text"])
        assert payload["harness_target"] == "sample_essay"
        assert payload["provider"] == "deepseek"
        assert payload["judge_provider"] == "openai"
        assert payload["results"][0]["scores"]["overall"] == 7.0
        mock_batch.assert_called_once()

    def test_harness_compare_sample_essay_tool_uses_fixed_judge_compare(self):
        server = IeltsMCPServer()

        def fake_pipeline(provider):
            def pipeline(question):
                return f"{provider} essay"

            return pipeline

        deepseek_results = [{
            "test_case_id": "mcp_manual_001",
            "user_input": "Question?",
            "scores": {"overall": 7.0},
            "error": None,
            "error_labels": [],
        }]
        openai_results = [{
            "test_case_id": "mcp_manual_001",
            "user_input": "Question?",
            "scores": {"overall": 6.0},
            "error": None,
            "error_labels": [],
        }]

        with patch("src.mcp_server.LLMFactory.create", return_value=Mock(model_name="judge")), \
             patch("src.mcp_server._build_generation_pipeline", side_effect=fake_pipeline), \
             patch("src.mcp_server.Evaluator.batch_evaluate", side_effect=[deepseek_results, openai_results]):
            response = server.handle(
                {
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tools/call",
                    "params": {
                        "name": "harness_compare_sample_essay",
                        "arguments": {
                            "question": "Should governments invest in public transport?",
                            "provider_a": "deepseek",
                            "provider_b": "openai",
                            "judge_provider": "openai",
                        },
                    },
                }
            )

        payload = json.loads(response["result"]["content"][0]["text"])
        assert payload["harness_target"] == "sample_essay"
        assert payload["comparison"]["overall_winner"] == "deepseek"
        assert payload["comparison"]["judge_provider"] == "openai"
        assert payload["provider_a_results"][0]["scores"]["overall"] == 7.0
        assert payload["provider_b_results"][0]["scores"]["overall"] == 6.0

    def test_unknown_tool_returns_error(self):
        server = IeltsMCPServer()
        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "unknown", "arguments": {}},
            }
        )

        assert response["error"]["code"] == -32602
        assert "Unknown tool" in response["error"]["message"]

    def test_missing_required_argument_returns_error(self):
        server = IeltsMCPServer()
        response = server.handle(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "harness_rule_evaluate", "arguments": {"question": "Q"}},
            }
        )

        assert response["error"]["code"] == -32602
        assert "output" in response["error"]["message"]
