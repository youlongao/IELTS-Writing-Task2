"""Streamlit app for the IELTS Task 2 idea and outline trainer."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.state import Argument, QuestionAnalysis, Scenario, WorkflowState
from src.agent.workflow import AgentWorkflow
from src.harness.evaluator import Evaluator
from src.llm.factory import LLMFactory
from src.prompts.templates import PromptManager
from src.rag.embedder import Embedder
from src.rag.retriever import Retriever
from src.rag.store import VectorStore
from src.utils.config import Config
from src.utils.logger import setup_logger


def z(text: str) -> str:
    """Decode ASCII unicode escapes so source files stay encoding-safe."""
    return text.encode("ascii").decode("unicode_escape")


logger = setup_logger("ielts_web")
DATA_DIR = Path(__file__).parent.parent / "data"
HISTORY_FILE = DATA_DIR / "user_history.json"
APP_VERSION = "2026-07-18-openai-deepseek-harness-v1"


st.set_page_config(
    page_title=z("IELTS Task 2 \\u5199\\u4f5c\\u601d\\u8def\\u8bad\\u7ec3\\u5668"),
    page_icon="IT",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .essay-box {
        background: #fff8e1;
        border: 1px solid #ffc107;
        padding: 1.5em;
        margin: 1em 0;
        border-radius: 8px;
        font-size: 1.04em;
        line-height: 1.8;
    }
    .essay-box p {
        margin: 0 0 1rem 0;
    }
    .essay-box p:last-child {
        margin-bottom: 0;
    }
    .reference-note {
        color: #475569;
        font-size: 0.92rem;
    }
    .study-route {
        background: #eef6ff;
        border: 1px solid #60a5fa;
        border-left: 6px solid #2563eb;
        padding: 1rem 1.1rem;
        border-radius: 8px;
        margin: 0.8rem 0 1.2rem 0;
        color: #0f172a;
        font-size: 1.02rem;
        line-height: 1.75;
    }
</style>
""",
    unsafe_allow_html=True,
)


DEFAULTS = {
    "initialized": False,
    "llm": None,
    "retriever": None,
    "workflow": None,
    "analysis": None,
    "multi_stance_text": "",
    "outline_text": "",
    "essay_text": "",
    "selected_stance": "",
    "question": "",
    "error_msg": "",
    "history": [],
    "outline_loading": False,
    "essay_loading": False,
    "rag_context": "",
    "rag_raw_results": {},
    "rag_counts": {},
    "user_essay_text": "",
    "user_essay_feedback": "",
    "essay_feedback_loading": False,
}

COLLECTION_LABELS = {
    "structures": z("\\u4f5c\\u6587\\u7ed3\\u6784\\u5e93"),
    "topics": z("\\u9ad8\\u9891\\u8bdd\\u9898\\u89c2\\u70b9\\u5e93"),
    "vocabulary": z("Band 7 \\u8bcd\\u4f19\\u8868\\u8fbe\\u5e93"),
    "rubrics": z("IELTS \\u8bc4\\u5206\\u6807\\u51c6\\u5e93"),
    "sample_essays": z("\\u8303\\u6587\\u7ed3\\u6784\\u53c2\\u8003\\u5e93"),
}

TOPIC_LABELS = {
    "education": z("\\u6559\\u80b2"),
    "technology": z("\\u79d1\\u6280"),
    "environment": z("\\u73af\\u5883"),
    "crime": z("\\u72af\\u7f6a"),
    "health": z("\\u5065\\u5eb7"),
    "transport": z("\\u4ea4\\u901a"),
    "work": z("\\u5de5\\u4f5c"),
    "globalization": z("\\u5168\\u7403\\u5316"),
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.get("app_version") != APP_VERSION:
    st.session_state.initialized = False
    st.session_state.llm = None
    st.session_state.retriever = None
    st.session_state.workflow = None
    st.session_state.rag_counts = {}
    st.session_state.app_version = APP_VERSION


def load_history():
    if not HISTORY_FILE.exists():
        return
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            st.session_state.history = json.load(f)
    except Exception:
        st.session_state.history = []


def save_history():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.history, f, ensure_ascii=False, indent=2)


def save_session_to_history():
    question = st.session_state.question
    if not question:
        return

    analysis_dict = None
    if st.session_state.analysis:
        analysis = st.session_state.analysis
        analysis_dict = {
            "question_type": analysis.question_type,
            "question_type_zh": analysis.question_type_zh,
            "question_type_en": analysis.question_type_en,
            "question_subtype": analysis.question_subtype,
            "question_subtype_zh": analysis.question_subtype_zh,
            "controversy": analysis.controversy,
            "stance_recommendations": analysis.stance_recommendations,
            "keywords": analysis.keywords,
            "topics": analysis.topics,
        }

    session_data = {
        "question": question,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "analysis": analysis_dict,
        "multi_stance_text": st.session_state.multi_stance_text,
        "outline_text": st.session_state.outline_text,
        "essay_text": st.session_state.essay_text,
        "selected_stance": st.session_state.selected_stance,
        "rag_context": st.session_state.rag_context,
        "user_essay_text": st.session_state.user_essay_text,
        "user_essay_feedback": st.session_state.user_essay_feedback,
    }

    for index, item in enumerate(st.session_state.history):
        if item["question"] == question:
            st.session_state.history[index] = session_data
            st.session_state.history.insert(0, st.session_state.history.pop(index))
            save_history()
            return

    st.session_state.history.insert(0, session_data)
    st.session_state.history = st.session_state.history[:50]
    save_history()


def restore_session(session_data: dict):
    st.session_state.question = session_data["question"]
    st.session_state.multi_stance_text = session_data.get("multi_stance_text", "")
    st.session_state.outline_text = session_data.get("outline_text", "")
    st.session_state.essay_text = session_data.get("essay_text", "")
    st.session_state.selected_stance = session_data.get("selected_stance", "")
    st.session_state.rag_context = session_data.get("rag_context", "")
    st.session_state.user_essay_text = session_data.get("user_essay_text", "")
    st.session_state.user_essay_feedback = session_data.get("user_essay_feedback", "")
    st.session_state.error_msg = ""
    st.session_state.outline_loading = False
    st.session_state.essay_loading = False
    st.session_state.essay_feedback_loading = False

    analysis_dict = session_data.get("analysis")
    if analysis_dict:
        st.session_state.analysis = QuestionAnalysis(
            question_type=analysis_dict.get("question_type", ""),
            question_type_zh=analysis_dict.get("question_type_zh", ""),
            question_type_en=analysis_dict.get("question_type_en", ""),
            question_subtype=analysis_dict.get("question_subtype", ""),
            question_subtype_zh=analysis_dict.get("question_subtype_zh", ""),
            controversy=analysis_dict.get("controversy", ""),
            stance_recommendations=analysis_dict.get("stance_recommendations", []),
            keywords=analysis_dict.get("keywords", []),
            topics=analysis_dict.get("topics", []),
        )
    else:
        st.session_state.analysis = None


if "history_loaded" not in st.session_state:
    load_history()
    st.session_state.history_loaded = True


def count_reference_store(store: VectorStore) -> dict:
    counts = {}
    for collection_name in store.COLLECTIONS:
        try:
            counts[collection_name] = store.count(collection_name)
        except Exception:
            counts[collection_name] = 0
    return counts


def init_engine(provider: str):
    config = Config()
    try:
        llm = LLMFactory.create(provider, config)
    except ValueError as exc:
        st.error(z("LLM \\u521d\\u59cb\\u5316\\u5931\\u8d25") + f":{exc}")
        return False

    st.session_state.llm = llm
    try:
        embedder = Embedder(config.get("rag.embedding_model", "all-MiniLM-L6-v2"))
        store = VectorStore(config.chroma_persist_dir)
        st.session_state.retriever = Retriever(embedder, store)
        st.session_state.rag_counts = count_reference_store(store)
    except Exception as exc:
        logger.warning("Reference store initialization failed: %s", exc)
        st.session_state.retriever = None
        st.session_state.rag_counts = {}

    st.session_state.workflow = AgentWorkflow(
        llm=llm,
        retriever=st.session_state.retriever,
        prompt_manager=PromptManager(),
    )
    st.session_state.initialized = True
    return True


def reset_work():
    for key in [
        "analysis",
        "multi_stance_text",
        "outline_text",
        "essay_text",
        "selected_stance",
        "error_msg",
        "rag_context",
        "rag_raw_results",
        "user_essay_text",
        "user_essay_feedback",
    ]:
        st.session_state[key] = None if key == "analysis" else ""
    st.session_state.outline_loading = False
    st.session_state.essay_loading = False
    st.session_state.essay_feedback_loading = False


def run_pipeline(question: str):
    reset_work()
    st.session_state.question = question

    try:
        workflow = st.session_state.workflow
        state = WorkflowState(user_input=question, scenario=Scenario.GENERATE)

        with st.spinner(z("\\u6b63\\u5728\\u5206\\u6790\\u9898\\u578b\\u3001\\u5173\\u952e\\u8bcd\\u548c\\u6838\\u5fc3\\u4e89\\u8bae\\u70b9...")):
            state = workflow._step_1_analyze_question(state)
            st.session_state.analysis = state.analysis

        if state.has_errors():
            st.session_state.error_msg = "; ".join(state.errors)
            return

        with st.spinner(z("\\u6b63\\u5728\\u5339\\u914d\\u8bdd\\u9898\\u89c2\\u70b9\\u3001\\u4f5c\\u6587\\u7ed3\\u6784\\u548c\\u8bcd\\u4f19\\u8868\\u8fbe...")):
            state = workflow._step_2_retrieve_knowledge(state)
            st.session_state.rag_context = state.rag_context
            st.session_state.rag_raw_results = state.rag_raw_results

        with st.spinner(z("\\u6b63\\u5728\\u751f\\u6210\\u53ef\\u5199\\u89c2\\u70b9\\u548c\\u8bba\\u8bc1\\u94fe...")):
            state = workflow._step_3_generate_ideas(state)

        if state.has_errors():
            st.session_state.error_msg = "; ".join(state.errors)
            return

        if state.arguments and state.arguments[0].main_idea_en:
            st.session_state.multi_stance_text = state.arguments[0].main_idea_en
            save_session_to_history()
            return

        st.session_state.error_msg = z("\\u6a21\\u578b\\u8fd4\\u56de\\u4e3a\\u7a7a\\uff0c\\u8bf7\\u91cd\\u8bd5\\u3002")
    except Exception as exc:
        st.session_state.error_msg = z("\\u5904\\u7406\\u5931\\u8d25") + f":{exc}"


def generate_outline(stance_label: str) -> bool:
    state = WorkflowState(
        user_input=st.session_state.question,
        scenario=Scenario.GENERATE,
        selected_stance=stance_label,
    )
    state.analysis = st.session_state.analysis
    state.rag_context = st.session_state.rag_context
    state.rag_raw_results = st.session_state.rag_raw_results or {}
    state.arguments = [Argument(number=1, main_idea_en=st.session_state.multi_stance_text)]

    state = st.session_state.workflow._step_4_build_outline(state)
    if state.outline and state.outline.tips:
        st.session_state.outline_text = state.outline.tips[0]
        return True
    if state.errors:
        st.session_state.error_msg = "; ".join(state.errors)
    return False


def generate_essay_text() -> str:
    workflow = st.session_state.workflow
    prompt = workflow.prompts.get("essay_generator", "v1")
    analysis = st.session_state.analysis
    rendered = prompt.render(
        question=st.session_state.question,
        question_type_zh=analysis.question_type_zh if analysis else "",
        question_type_en=analysis.question_type_en if analysis else "",
        stance=st.session_state.selected_stance,
        outline=st.session_state.outline_text,
        arguments=st.session_state.multi_stance_text,
        rag_context=st.session_state.rag_context,
    )
    return workflow.llm.generate(
        rendered["user"],
        system_message=rendered["system"],
        temperature=0.7,
        max_tokens=4096,
    )


def evaluate_user_essay_text(essay_text: str) -> str:
    if not essay_text.strip():
        return ""
    return st.session_state.workflow.evaluate_essay(
        question=st.session_state.question or "IELTS Task 2 question",
        essay=essay_text,
        outline=st.session_state.outline_text,
        rag_context=st.session_state.rag_context,
    )


def parse_stance_labels(text: str) -> list:
    labels = []
    patterns = [
        r"##\s*(Agree|Disagree|Partly Agree|Balanced|View A|View B|Own View).*?(?:\n|$)",
        r"##\s*Stance\s*(\d+).*?(?:\n|$)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            label = match.group(1)
            labels.append((str(label), match.group(0).strip("# \n")))
    return labels


def writing_path_config(question_type: str, question: str) -> dict:
    """Return learner-facing writing-path options for each Task 2 type."""
    qtype = question_type or "opinion"
    question_lower = question.lower()

    if qtype == "discussion":
        direct_discussion = any(
            pattern in question_lower
            for pattern in [
                "how has",
                "how have",
                "why has",
                "why have",
                "what has made",
                "what have made",
                "what are the reasons for",
                "what are the effects of",
                "what effects",
            ]
        )
        if direct_discussion:
            return {
                "heading": z("\\u786e\\u8ba4\\u89e3\\u91ca\\u578b\\u8ba8\\u8bba\\u8def\\u5f84") + " (Explanation Path)",
                "caption": z("\\u8fd9\\u7c7b\\u9898\\u4e0d\\u662f\\u540c\\u610f/\\u4e0d\\u540c\\u610f\\uff0c\\u4e5f\\u4e0d\\u4e00\\u5b9a\\u6709\\u89c2\\u70b9\\u5bf9\\u649e\\u3002\\u5173\\u952e\\u662f\\u89e3\\u91ca\\u73b0\\u8c61\\u5982\\u4f55\\u5f62\\u6210\\uff0c\\u5e76\\u628a\\u539f\\u56e0/\\u673a\\u5236/\\u5f71\\u54cd\\u8bf4\\u6210\\u95ed\\u73af\\u3002"),
                "label": z("\\u89e3\\u91ca\\u8def\\u5f84"),
                "custom_label": z("\\u4e5f\\u53ef\\u4ee5\\u8f93\\u5165\\u4f60\\u7684\\u89e3\\u91ca\\u8def\\u5f84"),
                "options": [
                    z("\\u5148\\u89e3\\u91ca\\u4f20\\u64ad\\u6e20\\u9053\\uff0c\\u518d\\u89e3\\u91ca\\u6d88\\u8d39\\u548c\\u6a21\\u4eff\\u673a\\u5236"),
                    z("\\u5148\\u5199\\u5a92\\u4f53/\\u4e92\\u8054\\u7f51\\u5982\\u4f55\\u6269\\u6563\\u8d8b\\u52bf\\uff0c\\u518d\\u5199\\u54c1\\u724c/\\u5e02\\u573a\\u5982\\u4f55\\u5f71\\u54cd\\u4e2a\\u4eba\\u9009\\u62e9"),
                    z("\\u6309\\u539f\\u56e0\\u94fe\\u7ec4\\u7ec7\\uff1a\\u5168\\u7403\\u66dd\\u5149 -> \\u793e\\u4f1a\\u8ba4\\u540c -> \\u8d2d\\u4e70\\u4fbf\\u5229 -> \\u65e5\\u5e38\\u7a7f\\u7740\\u53d8\\u5316"),
                ],
            }
        return {
            "heading": z("\\u786e\\u8ba4\\u8ba8\\u8bba\\u8def\\u5f84") + " (Discussion Path)",
            "caption": z("\\u8ba8\\u8bba\\u7c7b\\u9898\\u7684\\u91cd\\u70b9\\u4e0d\\u662f\\u7b80\\u5355\\u7ad9\\u961f\\uff0c\\u800c\\u662f\\u56de\\u5e94 both views \\u5e76\\u7ed9\\u51fa\\u660e\\u786e own opinion\\u3002"),
            "label": z("\\u8ba8\\u8bba\\u8def\\u5f84"),
            "custom_label": z("\\u4e5f\\u53ef\\u4ee5\\u8f93\\u5165\\u4f60\\u7684\\u8ba8\\u8bba\\u8def\\u5f84"),
            "options": [
                z("\\u5148\\u8ba8\\u8bba\\u89c2\\u70b9 A\\uff0c\\u518d\\u8ba8\\u8bba\\u89c2\\u70b9 B\\uff0c\\u6700\\u540e\\u7ed9\\u51fa\\u4e2a\\u4eba\\u503e\\u5411"),
                z("\\u5148\\u627f\\u8ba4 A \\u7684\\u5408\\u7406\\u6027\\uff0c\\u518d\\u91cd\\u70b9\\u652f\\u6301 B"),
                z("\\u5e73\\u8861\\u8ba8\\u8bba\\uff0c\\u4f46\\u7ed3\\u8bba\\u660e\\u786e\\u504f\\u5411\\u4e00\\u65b9"),
            ],
        }

    if qtype == "advantages_disadvantages":
        if "outweigh" in question_lower:
            return {
                "heading": z("\\u786e\\u8ba4\\u6743\\u8861\\u65b9\\u5411") + " (Weighing Direction)",
                "caption": z("\\u8fd9\\u7c7b\\u9898\\u9700\\u8981\\u5224\\u65ad\\u54ea\\u4e00\\u8fb9\\u66f4\\u91cd\\u8981\\uff0c\\u800c\\u4e0d\\u53ea\\u662f\\u5217\\u51fa\\u5229\\u5f0a\\u3002"),
                "label": z("\\u6743\\u8861\\u65b9\\u5411"),
                "custom_label": z("\\u4e5f\\u53ef\\u4ee5\\u8f93\\u5165\\u4f60\\u7684\\u6743\\u8861\\u65b9\\u5411"),
                "options": [
                    z("\\u5229\\u5927\\u4e8e\\u5f0a"),
                    z("\\u5f0a\\u5927\\u4e8e\\u5229"),
                    z("\\u6761\\u4ef6\\u6027\\u5224\\u65ad\\uff1a\\u5728\\u7279\\u5b9a\\u8303\\u56f4\\u5185\\u5229\\u5927\\u4e8e\\u5f0a\\u6216\\u5f0a\\u5927\\u4e8e\\u5229"),
                ],
            }
        return {
            "heading": z("\\u786e\\u8ba4\\u5c55\\u5f00\\u7ed3\\u6784") + " (Development Structure)",
            "caption": z("\\u5982\\u679c\\u9898\\u76ee\\u53ea\\u8981\\u6c42\\u8ba8\\u8bba advantages and disadvantages\\uff0c\\u6838\\u5fc3\\u662f\\u628a\\u4e24\\u9762\\u8bf4\\u6e05\\u695a\\uff0c\\u4e0d\\u5fc5\\u786c\\u505a\\u4e00\\u4e2a outweigh \\u5224\\u65ad\\u3002"),
            "label": z("\\u5c55\\u5f00\\u7ed3\\u6784"),
            "custom_label": z("\\u4e5f\\u53ef\\u4ee5\\u8f93\\u5165\\u4f60\\u7684\\u5c55\\u5f00\\u7ed3\\u6784"),
            "options": [
                z("Body 1 \\u5199 advantages\\uff0cBody 2 \\u5199 disadvantages"),
                z("Body 1 \\u5199\\u4e3b\\u8981\\u597d\\u5904\\uff0cBody 2 \\u5199\\u4e3b\\u8981\\u95ee\\u9898\\u4e0e\\u9650\\u5236"),
            ],
        }

    if qtype == "problem_solution":
        return {
            "heading": z("\\u786e\\u8ba4\\u95ee\\u9898-\\u89e3\\u51b3\\u6620\\u5c04") + " (Problem-Solution Mapping)",
            "caption": z("\\u95ee\\u9898\\u89e3\\u51b3\\u7c7b\\u9898\\u7684\\u5173\\u952e\\u662f\\u95ed\\u73af\\uff1a\\u63d0\\u51fa\\u7684 solution \\u5fc5\\u987b\\u76f4\\u63a5\\u56de\\u5e94 problem \\u6216 cause\\u3002"),
            "label": z("\\u95ed\\u73af\\u8def\\u5f84"),
            "custom_label": z("\\u4e5f\\u53ef\\u4ee5\\u8f93\\u5165\\u4f60\\u7684\\u95ee\\u9898-\\u89e3\\u51b3\\u8def\\u5f84"),
            "options": [
                z("\\u95ee\\u9898 1 + \\u5bf9\\u5e94\\u89e3\\u51b3\\u65b9\\u6848 1\\uff1b\\u95ee\\u9898 2 + \\u5bf9\\u5e94\\u89e3\\u51b3\\u65b9\\u6848 2"),
                z("\\u5148\\u5206\\u6790\\u539f\\u56e0\\uff0c\\u518d\\u63d0\\u51fa\\u5339\\u914d\\u63aa\\u65bd"),
                z("\\u6309\\u5f71\\u54cd\\u5bf9\\u8c61\\u7ec4\\u7ec7\\uff1a\\u4e2a\\u4eba / \\u793e\\u4f1a / \\u653f\\u5e9c\\u5404\\u81ea\\u7684\\u95ee\\u9898\\u4e0e\\u63aa\\u65bd"),
            ],
        }

    if qtype in {"two_part", "two_part_question"}:
        return {
            "heading": z("\\u786e\\u8ba4\\u53cc\\u95ee\\u9898\\u56de\\u7b54\\u987a\\u5e8f") + " (Two-Part Answer Order)",
            "caption": z("\\u53cc\\u95ee\\u9898\\u7684\\u91cd\\u70b9\\u662f\\u4e24\\u4e2a\\u95ee\\u9898\\u90fd\\u8981\\u76f4\\u63a5\\u56de\\u7b54\\uff0c\\u4e0d\\u8981\\u628a\\u5176\\u4e2d\\u4e00\\u4e2a\\u5199\\u6210\\u80cc\\u666f\\u4ecb\\u7ecd\\u3002"),
            "label": z("\\u56de\\u7b54\\u987a\\u5e8f"),
            "custom_label": z("\\u4e5f\\u53ef\\u4ee5\\u8f93\\u5165\\u4f60\\u7684\\u53cc\\u95ee\\u9898\\u56de\\u7b54\\u987a\\u5e8f"),
            "options": [
                z("Q1 \\u4e00\\u4e2a\\u4e3b\\u4f53\\u6bb5\\uff1bQ2 \\u4e00\\u4e2a\\u4e3b\\u4f53\\u6bb5"),
                z("\\u6bcf\\u6bb5\\u90fd\\u76f4\\u63a5\\u56de\\u7b54\\u95ee\\u9898 + \\u539f\\u56e0 + \\u4f8b\\u5b50"),
                z("\\u5148\\u56de\\u7b54\\u66f4\\u76f4\\u63a5\\u7684\\u95ee\\u9898\\uff0c\\u518d\\u56de\\u7b54\\u9700\\u8981\\u89e3\\u91ca\\u7684\\u95ee\\u9898"),
            ],
        }

    if qtype == "opinion" and "negative trend" in question_lower:
        return {
            "heading": z("\\u786e\\u8ba4\\u8d1f\\u9762\\u8d8b\\u52bf\\u5224\\u65ad") + " (Negative Trend Judgement)",
            "caption": z("\\u8fd9\\u7c7b\\u9898\\u4e0d\\u662f\\u666e\\u901a agree/disagree\\uff0c\\u800c\\u662f\\u8981\\u5224\\u65ad\\u8fd9\\u4e2a\\u53d8\\u5316\\u5728\\u591a\\u5927\\u7a0b\\u5ea6\\u4e0a\\u662f\\u8d1f\\u9762\\u8d8b\\u52bf\\u3002"),
            "label": z("\\u8d1f\\u9762\\u7a0b\\u5ea6\\u5224\\u65ad"),
            "custom_label": z("\\u4e5f\\u53ef\\u4ee5\\u8f93\\u5165\\u4f60\\u7684\\u8d1f\\u9762\\u8d8b\\u52bf\\u5224\\u65ad"),
            "options": [
                z("\\u4e3b\\u8981\\u662f\\u8d1f\\u9762\\u8d8b\\u52bf\\uff1a\\u5546\\u4e1a\\u5316\\u4f11\\u95f2\\u548c\\u8fc7\\u5ea6\\u6d88\\u8d39\\u95ee\\u9898\\u66f4\\u7a81\\u51fa"),
                z("\\u6709\\u4e00\\u5b9a\\u8d1f\\u9762\\uff0c\\u4f46\\u4e0d\\u5b8c\\u5168\\u8d1f\\u9762\\uff1a\\u627f\\u8ba4\\u793e\\u4ea4\\u4ef7\\u503c\\uff0c\\u4f46\\u6279\\u5224\\u8fc7\\u5ea6\\u4f9d\\u8d56\\u5546\\u4e1a\\u7a7a\\u95f4"),
                z("\\u4e0d\\u5b9c\\u7b80\\u5355\\u5b9a\\u6027\\u4e3a\\u8d1f\\u9762\\uff1a\\u8d2d\\u7269\\u4e2d\\u5fc3\\u4e5f\\u627f\\u62c5\\u4e86\\u73b0\\u4ee3\\u793e\\u4ea4\\u7a7a\\u95f4\\u7684\\u529f\\u80fd"),
            ],
        }

    if qtype == "opinion" and "positive or negative development" in question_lower:
        return {
            "heading": z("\\u786e\\u8ba4\\u6b63\\u8d1f\\u9762\\u53d1\\u5c55\\u5224\\u65ad") + " (Positive/Negative Development Judgement)",
            "caption": z("\\u8fd9\\u7c7b\\u9898\\u8981\\u5224\\u65ad\\u8fd9\\u4e2a\\u53d8\\u5316\\u603b\\u4f53\\u4e0a\\u662f positive \\u8fd8\\u662f negative\\uff0c\\u5e76\\u8bf4\\u660e\\u539f\\u56e0\\u3002"),
            "label": z("\\u6b63\\u8d1f\\u9762\\u5224\\u65ad"),
            "custom_label": z("\\u4e5f\\u53ef\\u4ee5\\u8f93\\u5165\\u4f60\\u7684\\u6b63\\u8d1f\\u9762\\u5224\\u65ad"),
            "options": [
                z("\\u603b\\u4f53\\u662f\\u6b63\\u9762\\u53d1\\u5c55"),
                z("\\u603b\\u4f53\\u662f\\u8d1f\\u9762\\u53d1\\u5c55"),
                z("\\u6709\\u6b63\\u8d1f\\u4e24\\u9762\\uff0c\\u4f46\\u660e\\u786e\\u504f\\u5411\\u4e00\\u8fb9"),
            ],
        }

    return {
        "heading": z("\\u786e\\u8ba4\\u5199\\u4f5c\\u7acb\\u573a") + " (Writing Position)",
        "caption": z("\\u89c2\\u70b9\\u7c7b\\u9898\\u9700\\u8981\\u6709\\u660e\\u786e overall position\\uff0c\\u4f46\\u4e0d\\u4e00\\u5b9a\\u662f\\u7edd\\u5bf9\\u540c\\u610f\\u6216\\u7edd\\u5bf9\\u4e0d\\u540c\\u610f\\u3002"),
        "label": z("\\u7acb\\u573a"),
        "custom_label": z("\\u4e5f\\u53ef\\u4ee5\\u8f93\\u5165\\u81ea\\u5df1\\u7684\\u7acb\\u573a"),
        "options": [
            z("\\u540c\\u610f (Agree)"),
            z("\\u4e0d\\u540c\\u610f (Disagree)"),
            z("\\u90e8\\u5206\\u540c\\u610f\\uff0c\\u4f46\\u503e\\u5411\\u4e00\\u8fb9 (Partly agree with a clear leaning)"),
        ],
    }


def format_essay(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    has_blank_lines = bool(re.search(r"\n\s*\n", text))
    starters = (
        "One major reason",
        "Another reason",
        "On the one hand",
        "On the other hand",
        "I see this",
        "In conclusion",
        "To conclude",
        "Overall",
    )
    starter_pattern = "|".join(re.escape(starter) for starter in starters)
    text = re.sub(rf"(?<=[.!?])\s*\n\s*(?=(?:{starter_pattern})\b)", "\n\n", text)
    if not has_blank_lines:
        text = re.sub(r"\.\s*\n(?=[A-Z])", ".\n\n", text)
        text = re.sub(r"\n(?=[A-Z][a-z])", "\n\n", text)
    paragraphs = re.split(r"\n\s*\n", text)
    return "\n\n".join(
        " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        for paragraph in paragraphs
        if paragraph.strip()
    )


def essay_to_html(text: str) -> str:
    paragraphs = [
        paragraph.strip()
        for paragraph in format_essay(text).split("\n\n")
        if paragraph.strip()
    ]
    escaped = [
        paragraph.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for paragraph in paragraphs
    ]
    return "".join(f"<p>{paragraph}</p>" for paragraph in escaped)


def extract_section(text: str, titles: list[str]) -> str:
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        heading = line.strip().lstrip("#").strip().lower()
        if any(title.lower() in heading for title in titles):
            start = index
            break
    if start is None:
        return ""

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("## "):
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def strip_leading_markdown_heading(text: str) -> str:
    """Remove a duplicated section heading from extracted markdown."""
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].lstrip().startswith("#"):
        lines.pop(0)
    return "\n".join(lines).strip()


def remove_sections(text: str, titles: list[str]) -> str:
    lines = text.splitlines()
    output = []
    skipping = False
    for line in lines:
        if line.startswith("## "):
            heading = line.strip().lstrip("#").strip().lower()
            skipping = any(title.lower() in heading for title in titles)
        if not skipping:
            output.append(line)
    return "\n".join(output).strip()


with st.sidebar:
    st.title(z("\\u8bbe\\u7f6e") + " (Settings)")
    provider = st.selectbox(
        z("LLM \\u63d0\\u4f9b\\u5546"),
        options=["openai", "deepseek"],
        index=0,
        format_func=lambda value: {
            "openai": "OpenAI",
            "deepseek": "DeepSeek",
        }.get(value, value),
    )

    if st.button(z("\\u8fde\\u63a5"), type="primary", use_container_width=True):
        if init_engine(provider):
            st.success(z("\\u5df2\\u8fde\\u63a5") + f":{provider}")

    if st.session_state.initialized:
        st.success(z("\\u5f15\\u64ce\\u5c31\\u7eea"))
    else:
        st.info(z("\\u8bf7\\u5148\\u8fde\\u63a5\\u6a21\\u578b"))

    with st.expander(z("\\u8d44\\u6599\\u5e93\\u72b6\\u6001"), expanded=False):
        counts = st.session_state.rag_counts or {}
        if counts:
            for name, count in counts.items():
                st.write(f"{COLLECTION_LABELS.get(name, name)}: {count}")
        else:
            st.caption(z("\\u5c1a\\u672a\\u8fde\\u63a5\\uff0c\\u6216\\u8d44\\u6599\\u5e93\\u8fd8\\u6ca1\\u6709\\u6784\\u5efa\\u3002"))
        st.caption(z("\\u5b66\\u751f\\u4fa7\\u4f1a\\u628a\\u5b83\\u611f\\u77e5\\u4e3a\\u8bdd\\u9898\\u89c2\\u70b9\\u5e93\\u3001\\u8bcd\\u4f19\\u8868\\u8fbe\\u5e93\\u548c\\u7ed3\\u6784\\u53c2\\u8003\\u5e93\\u3002"))

    st.divider()
    st.subheader(z("\\u5386\\u53f2\\u8bb0\\u5f55") + " (History)")
    if st.button(z("\\u65b0\\u5efa\\u9898\\u76ee"), use_container_width=True):
        reset_work()
        st.session_state.question = ""
        st.rerun()

    if st.session_state.history:
        if st.button(z("\\u6e05\\u7a7a\\u5386\\u53f2"), use_container_width=True):
            st.session_state.history = []
            save_history()
            st.rerun()

        for index, item in enumerate(st.session_state.history):
            label = item["question"][:45] + ("..." if len(item["question"]) > 45 else "")
            if st.button(f"{label} ({item['time']})", key=f"hist_{index}", use_container_width=True):
                restore_session(item)
                st.rerun()
    else:
        st.caption(z("\\u6682\\u65e0\\u5386\\u53f2\\u8bb0\\u5f55"))


st.title(z("IELTS Task 2 \\u5199\\u4f5c\\u601d\\u8def\\u4e0e\\u8bba\\u8bc1\\u8bad\\u7ec3\\u5668"))
st.caption(z("\\u5148\\u7528\\u4e2d\\u6587\\u7406\\u89e3\\u903b\\u8f91\\uff0c\\u518d\\u5b66\\u4e60\\u5982\\u4f55\\u7528 IELTS \\u82f1\\u6587\\u8868\\u8fbe\\u3002"))
st.markdown(
    '<div class="study-route">'
    + z("\\u5b66\\u4e60\\u8def\\u5f84\\uff1a1. \\u5148\\u770b\\u5199\\u4f5c\\u63d0\\u793a\\uff1b2. \\u53ef\\u8f93\\u5165\\u81ea\\u5df1\\u7684\\u4f5c\\u6587\\u8bc4\\u5206\\uff0c\\u4e5f\\u53ef\\u76f4\\u63a5\\u770b\\u53c2\\u8003\\u8303\\u6587\\uff1b3. \\u518d\\u6839\\u636e Band \\u53cd\\u9988\\u4fee\\u6539\\u3002")
    + "</div>",
    unsafe_allow_html=True,
)

question = st.text_area(
    z("IELTS Task 2 \\u9898\\u76ee"),
    value=st.session_state.question,
    placeholder="Some people think governments should invest more in public transport rather than roads. To what extent do you agree or disagree?",
    height=110,
)

col_start, col_hint = st.columns([1, 4])
with col_start:
    if st.button(z("\\u751f\\u6210\\u601d\\u8def"), type="primary", use_container_width=True):
        if not st.session_state.initialized:
            st.error(z("\\u8bf7\\u5148\\u5728\\u5de6\\u4fa7\\u680f\\u8fde\\u63a5\\u6a21\\u578b\\u3002"))
        elif question.strip():
            run_pipeline(question.strip())
            st.rerun()
with col_hint:
    if not st.session_state.initialized:
        st.warning(z("\\u8fde\\u63a5\\u6a21\\u578b\\u540e\\u5373\\u53ef\\u5f00\\u59cb\\u3002"))

if st.session_state.error_msg:
    st.error(st.session_state.error_msg)

if st.session_state.analysis:
    st.divider()
    st.subheader(z("\\u9898\\u76ee\\u62c6\\u89e3") + " (Question Analysis)")
    analysis = st.session_state.analysis
    c1, c2, c3 = st.columns(3)
    type_label = analysis.question_type_zh or analysis.question_type_en or analysis.question_type
    if analysis.question_subtype_zh:
        type_label = f"{type_label} - {analysis.question_subtype_zh}"
    c1.metric(z("\\u9898\\u578b") + " (Question Type)", type_label)
    c2.metric(z("\\u9898\\u578b\\u4ee3\\u7801") + " (Type Code)", analysis.question_type)
    topics = ", ".join(TOPIC_LABELS.get(topic, topic) for topic in analysis.topics)
    c3.metric(z("\\u8bdd\\u9898") + " (Topic)", topics if topics else z("\\u81ea\\u52a8\\u8bc6\\u522b"))
    if analysis.controversy:
        st.markdown(analysis.controversy)

if st.session_state.multi_stance_text:
    st.divider()
    st.subheader(z("\\u53ef\\u5199\\u89c2\\u70b9\\u4e0e\\u8bba\\u8bc1\\u94fe") + " (Writable Ideas and Argument Chains)")
    st.markdown(
        '<p class="reference-note">'
        + z("\\u4e2d\\u6587\\u7528\\u4e8e\\u7406\\u89e3\\u903b\\u8f91\\uff0c\\u82f1\\u6587\\u7528\\u4e8e\\u8fc1\\u79fb\\u5230\\u4f5c\\u6587\\u8868\\u8fbe\\u3002")
        + "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(st.session_state.multi_stance_text)

    if not st.session_state.outline_text and not st.session_state.outline_loading:
        analysis = st.session_state.analysis
        config = writing_path_config(
            analysis.question_type if analysis else "opinion",
            st.session_state.question,
        )
        st.subheader(config["heading"])
        st.caption(config["caption"])
        choice = st.radio(config["label"], options=config["options"], index=None)
        custom_path = st.text_input(config["custom_label"])
        selected_path = custom_path.strip() or choice
        if st.button(z("\\u751f\\u6210\\u5efa\\u8bae\\u5927\\u7eb2"), type="primary"):
            if selected_path:
                st.session_state.selected_stance = selected_path
                st.session_state.outline_loading = True
                st.rerun()
            st.warning(z("\\u8bf7\\u5148\\u9009\\u62e9\\u6216\\u8f93\\u5165\\u4e00\\u4e2a\\u5199\\u4f5c\\u8def\\u5f84\\u3002"))

    if st.session_state.outline_loading and not st.session_state.outline_text:
        with st.spinner(z("\\u6b63\\u5728\\u6839\\u636e\\u4f60\\u7684\\u5199\\u4f5c\\u8def\\u5f84\\u751f\\u6210\\u56db\\u6bb5\\u5f0f\\u5927\\u7eb2...")):
            success = generate_outline(st.session_state.selected_stance)
        st.session_state.outline_loading = False
        if success:
            save_session_to_history()
        else:
            st.session_state.error_msg = st.session_state.error_msg or z("\\u5927\\u7eb2\\u751f\\u6210\\u5931\\u8d25\\uff0c\\u8bf7\\u91cd\\u8bd5\\u3002")
        st.rerun()

if st.session_state.outline_text:
    st.divider()
    writing_tip_titles = [z("\\u5199\\u4f5c\\u63d0\\u793a"), "Writing Tips"]
    hidden_titles = writing_tip_titles + [
        "Band 7 Readiness Checklist",
        z("Band 7 \\u51c6\\u5907\\u5ea6\\u68c0\\u67e5"),
    ]
    writing_tips = strip_leading_markdown_heading(
        extract_section(st.session_state.outline_text, writing_tip_titles)
    )
    outline_for_display = remove_sections(st.session_state.outline_text, hidden_titles)

    st.subheader(z("\\u5efa\\u8bae\\u5927\\u7eb2") + " (Suggested Outline)")
    st.markdown(outline_for_display)

    st.subheader(z("\\u5199\\u4f5c\\u63d0\\u793a") + " (Writing Tips)")
    st.markdown(
        '<div class="study-route">'
        + z("\\u8fd9\\u4e00\\u6b65\\u662f\\u5199\\u4f5c\\u524d\\u6700\\u91cd\\u8981\\u7684\\u63d0\\u9192\\uff1a\\u5148\\u770b\\u61c2\\u6bb5\\u843d\\u4efb\\u52a1\\uff0c\\u518d\\u52a8\\u7b14\\u5199\\u82f1\\u6587\\u3002")
        + "</div>",
        unsafe_allow_html=True,
    )
    if writing_tips:
        st.markdown(writing_tips)
    else:
        st.markdown(
            "- "
            + z("\\u6bcf\\u4e2a\\u4e3b\\u4f53\\u6bb5\\u53ea\\u505a\\u4e00\\u4ef6\\u4e8b\\uff1a\\u4e3b\\u89c2\\u70b9 -> \\u539f\\u56e0 -> \\u7ed3\\u679c -> \\u4f8b\\u5b50 -> \\u6263\\u9898\\u3002")
        )

    st.subheader(z("\\u5199\\u4f5c\\u7ec3\\u4e60\\u4e0e\\u53c2\\u8003\\u8303\\u6587") + " (Writing Practice and Sample Essay)")
    st.caption(z("\\u8fd9\\u4e24\\u6b65\\u6ca1\\u6709\\u5f3a\\u5236\\u987a\\u5e8f\\uff1a\\u4f60\\u53ef\\u4ee5\\u5148\\u63d0\\u4ea4\\u81ea\\u5df1\\u7684\\u4f5c\\u6587\\u8bc4\\u5206\\uff0c\\u4e5f\\u53ef\\u4ee5\\u76f4\\u63a5\\u67e5\\u770b\\u53c2\\u8003\\u8303\\u6587\\u3002"))
    essay_col, sample_col = st.columns(2)

    with essay_col:
        st.markdown("**" + z("\\u6211\\u7684\\u4f5c\\u6587\\u8bc4\\u5206") + " (My Essay Evaluation)**")
        user_essay = st.text_area(
            z("\\u7c98\\u8d34\\u4f60\\u7684\\u5b8c\\u6574\\u4f5c\\u6587"),
            value=st.session_state.user_essay_text,
            height=260,
            placeholder="Paste your IELTS Task 2 essay here...",
        )
        st.session_state.user_essay_text = user_essay
        if st.button(z("\\u63d0\\u4ea4\\u8bc4\\u5206"), type="primary", use_container_width=True):
            if not st.session_state.initialized:
                st.error(z("\\u8bf7\\u5148\\u8fde\\u63a5\\u6a21\\u578b\\u3002"))
            elif not user_essay.strip():
                st.warning(z("\\u8bf7\\u5148\\u7c98\\u8d34\\u4f60\\u7684\\u4f5c\\u6587\\u3002"))
            else:
                st.session_state.essay_feedback_loading = True
                st.rerun()

        if st.session_state.essay_feedback_loading:
            try:
                with st.spinner(z("\\u6b63\\u5728\\u6839\\u636e\\u5b9e\\u9645 IELTS \\u6807\\u51c6\\u8fdb\\u884c\\u601d\\u8def\\u8d28\\u91cf\\u548c Band \\u4f30\\u5206...")):
                    st.session_state.user_essay_feedback = evaluate_user_essay_text(st.session_state.user_essay_text)
                    save_session_to_history()
            except Exception as exc:
                st.session_state.error_msg = z("\\u4f5c\\u6587\\u8bc4\\u5206\\u5931\\u8d25") + f":{exc}"
            st.session_state.essay_feedback_loading = False
            st.rerun()

        st.markdown("**" + z("\\u5b9e\\u9645 Band \\u4f30\\u5206") + " (Actual Band Estimate)**")
        if st.session_state.user_essay_feedback:
            st.markdown(st.session_state.user_essay_feedback)
        else:
            st.info(z("\\u63d0\\u4ea4\\u540e\\uff0c\\u8bc4\\u5206\\u7ed3\\u679c\\u4f1a\\u76f4\\u63a5\\u663e\\u793a\\u5728\\u8fd9\\u91cc\\u3002"))

    with sample_col:
        st.markdown("**" + z("\\u53c2\\u8003\\u8303\\u6587") + " (Sample Essay)**")
        st.caption(z("\\u53ef\\u4ee5\\u76f4\\u63a5\\u67e5\\u770b\\uff0c\\u4e0d\\u9700\\u8981\\u5148\\u63d0\\u4ea4\\u81ea\\u5df1\\u7684\\u4f5c\\u6587\\u3002"))
        if not st.session_state.essay_text and not st.session_state.essay_loading:
            if st.button(z("\\u751f\\u6210\\u53c2\\u8003\\u8303\\u6587"), use_container_width=True):
                st.session_state.essay_loading = True
                st.rerun()

        if st.session_state.essay_loading and not st.session_state.essay_text:
            try:
                with st.spinner(z("\\u6b63\\u5728\\u751f\\u6210\\u53c2\\u8003\\u8303\\u6587...")):
                    st.session_state.essay_text = generate_essay_text()
                    save_session_to_history()
            except Exception as exc:
                st.session_state.error_msg = z("\\u53c2\\u8003\\u8303\\u6587\\u751f\\u6210\\u5931\\u8d25") + f":{exc}"
            st.session_state.essay_loading = False
            st.rerun()

        if st.session_state.essay_text:
            text = st.session_state.essay_text
            if "---" in text:
                essay_body, metadata = text.split("---", 1)
            else:
                essay_body, metadata = text, ""
            essay_body = format_essay(essay_body)
            st.markdown(f'<div class="essay-box">{essay_to_html(essay_body)}</div>', unsafe_allow_html=True)
            if metadata:
                st.markdown(metadata)
            st.download_button(
                z("\\u4e0b\\u8f7d\\u53c2\\u8003\\u8303\\u6587"),
                data=essay_body,
                file_name="ielts_sample_essay.txt",
                mime="text/plain",
            )

with st.expander(z("\\u5185\\u90e8\\u8d28\\u91cf\\u8bca\\u65ad") + " (Internal Quality Diagnostics)", expanded=False):
    st.caption(z("\\u4ec5\\u7528\\u4e8e\\u5f00\\u53d1\\u8c03\\u8bd5\\uff1b\\u5b66\\u751f\\u4e0d\\u9700\\u8981\\u7406\\u89e3\\u8fd9\\u4e00\\u5c42\\u3002"))
    if st.session_state.multi_stance_text:
        rule_scores = Evaluator.rule_evaluate(
            st.session_state.question,
            st.session_state.multi_stance_text,
        )
        st.json(rule_scores)
    if st.session_state.rag_raw_results:
        st.write({name: len(items) for name, items in st.session_state.rag_raw_results.items()})

st.divider()
st.caption(z("\\u771f\\u6b63\\u7684\\u63d0\\u5347\\u6765\\u81ea\\u7406\\u89e3\\u8bba\\u8bc1\\u903b\\u8f91\\uff0c\\u5e76\\u7528\\u81ea\\u5df1\\u7684\\u8bed\\u8a00\\u91cd\\u65b0\\u5199\\u51fa\\u6765\\u3002"))
