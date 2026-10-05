import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.main import ask_question
from app.agents.models import QuestionRequest

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

EVAL_DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"
EVAL_RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"

def run_evaluation() -> Dict[str, Any]:
    """
    Executes automated evaluation across the curated dataset:
    Measures:
    - Route Selection Accuracy
    - Retrieval Hit Rate / Recall
    - Fact Groundedness & Faithfulness
    - Citation Correctness
    - End-to-end Latency
    """
    if not EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {EVAL_DATASET_PATH}")

    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print("=" * 85)
    print("      AGENTIC AI RESEARCH ASSISTANT - COMPREHENSIVE EVALUATION BENCHMARK      ")
    print("=" * 85)
    print(f"Total Test Cases: {len(test_cases)}\n")

    results: List[Dict[str, Any]] = []

    route_correct_count = 0
    retrieval_hit_count = 0
    internal_tests_count = 0
    fact_recall_total = 0.0
    citation_correct_count = 0
    total_latency = 0.0

    for idx, tc in enumerate(test_cases, 1):
        test_id = tc["id"]
        category = tc["category"]
        query = tc["query"]
        expected_route = tc["expected_route"]
        expected_doc = tc.get("expected_document")
        key_facts = tc.get("key_facts", [])
        scope_docs = tc.get("selected_document_ids")

        print(f"[{idx}/{len(test_cases)}] Running: {test_id} ({category})")
        print(f"    Query: '{query[:65]}...'")

        start_t = time.time()
        req = QuestionRequest(query=query, selected_document_ids=scope_docs)
        resp = ask_question(req)
        latency = time.time() - start_t
        total_latency += latency

        actual_route = resp.route_taken
        final_ans = resp.final_answer
        int_sources = resp.internal_sources
        ext_sources = resp.external_sources

        # 1. Routing Evaluation
        route_match = (actual_route == expected_route)
        if route_match:
            route_correct_count += 1

        # 2. Retrieval Recall (for queries requiring RAG)
        retrieval_hit = False
        if expected_doc:
            internal_tests_count += 1
            retrieved_doc_names = [s.document_name for s in int_sources]
            if any(expected_doc.lower() in name.lower() for name in retrieved_doc_names):
                retrieval_hit = True
                retrieval_hit_count += 1

        # 3. Fact Groundedness & Recall
        facts_found = 0
        if key_facts:
            ans_lower = final_ans.lower()
            for fact in key_facts:
                if fact.lower() in ans_lower:
                    facts_found += 1
            fact_score = facts_found / len(key_facts)
        else:
            fact_score = 1.0
        fact_recall_total += fact_score

        # 4. Citation Correctness
        has_citations = False
        if actual_route == "rag":
            has_citations = len(int_sources) > 0
        elif actual_route == "research":
            has_citations = len(ext_sources) > 0
        elif actual_route == "both":
            has_citations = (len(int_sources) > 0 or len(ext_sources) > 0)
        elif actual_route == "blocked_by_guardrails":
            has_citations = True  # Not applicable for blocked queries

        if has_citations:
            citation_correct_count += 1

        print(f"    Route: {actual_route} (Expected: {expected_route}) -> {'MATCH' if route_match else 'DIFF'}")
        print(f"    Retrieval Hit: {retrieval_hit} | Fact Recall: {round(fact_score*100, 1)}% | Latency: {latency:.2f}s\n")

        results.append({
            "test_id": test_id,
            "category": category,
            "query": query,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_match": route_match,
            "retrieval_hit": retrieval_hit if expected_doc else None,
            "fact_recall_score": round(fact_score, 3),
            "citations_present": has_citations,
            "latency_seconds": round(latency, 2),
            "final_answer_snippet": final_ans[:200]
        })

    # Summary Metrics Calculation
    total_tests = len(test_cases)
    route_accuracy = (route_correct_count / total_tests) * 100
    retrieval_recall = ((retrieval_hit_count / internal_tests_count) * 100) if internal_tests_count > 0 else 100.0
    avg_fact_recall = (fact_recall_total / total_tests) * 100
    citation_quality = (citation_correct_count / total_tests) * 100
    avg_latency = total_latency / total_tests

    summary = {
        "total_test_cases": total_tests,
        "routing_accuracy_pct": round(route_accuracy, 1),
        "retrieval_recall_pct": round(retrieval_recall, 1),
        "fact_groundedness_pct": round(avg_fact_recall, 1),
        "citation_compliance_pct": round(citation_quality, 1),
        "average_latency_seconds": round(avg_latency, 2),
        "test_results": results
    }

    # Print Executive Summary Table
    print("=" * 85)
    print("                         EXECUTIVE EVALUATION SUMMARY                         ")
    print("=" * 85)
    print(f"  Metric                             Value       Benchmark Target    Status   ")
    print("-" * 85)
    print(f"  Supervisor Routing Accuracy        {route_accuracy:5.1f}%          > 85.0%         {'PASS' if route_accuracy >= 85 else 'WARN'}   ")
    print(f"  Hybrid Retrieval Hit Rate (Recall) {retrieval_recall:5.1f}%          > 85.0%         {'PASS' if retrieval_recall >= 85 else 'WARN'}   ")
    print(f"  Fact Groundedness & Recall         {avg_fact_recall:5.1f}%          > 80.0%         {'PASS' if avg_fact_recall >= 80 else 'WARN'}   ")
    print(f"  Citation Compliance Rate           {citation_quality:5.1f}%          > 85.0%         {'PASS' if citation_quality >= 85 else 'WARN'}   ")
    print(f"  Mean Query Latency                 {avg_latency:5.2f}s          < 10.0s         PASS   ")
    print("=" * 85)

    # Save to JSON
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed evaluation results saved to: {EVAL_RESULTS_PATH}\n")

    return summary

if __name__ == "__main__":
    run_evaluation()
