import unittest

try:
    from langchain_core.documents import Document
except ModuleNotFoundError:
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict

from cvrag.rag.evaluation import (
    RetrievalEvalCase,
    evaluate_retrieval,
    summarize_retrieval_results,
)
from cvrag.rag.retriever import build_query_profile, keyword_retrieve, rerank_documents


class MetadataRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(
                page_content="Led Kubernetes migration for Acme and reduced deployment time by 60% in 2023.",
                metadata={"section": "experience", "company": "Acme", "dates": "2023"},
            ),
            Document(
                page_content="Skills: AWS, Terraform, Docker, Python.",
                metadata={"section": "skills"},
            ),
            Document(
                page_content="Built a streaming API platform project for internal services.",
                metadata={"section": "projects", "name": "Streaming Platform"},
            ),
            Document(
                page_content="BSc Computer Science.",
                metadata={"section": "education"},
            ),
        ]

    def test_keyword_retrieve_prefers_section_and_metrics(self):
        profile = build_query_profile(
            "What quantified impact did I have in my Kubernetes experience at Acme in 2023?"
        )

        results = keyword_retrieve(self.documents, profile, limit=3)

        self.assertEqual(results[0].metadata["section"], "experience")
        self.assertIn("60%", results[0].page_content)

    def test_rerank_promotes_skills_for_skills_query(self):
        profile = build_query_profile("Which cloud skills and tools do I have?")
        ranked = rerank_documents(list(reversed(self.documents)), profile)

        self.assertEqual(ranked[0].metadata["section"], "skills")

    def test_eval_suite_reports_pass_fail_summary(self):
        class StubRetriever:
            def __init__(self, documents):
                self.documents = documents

            def invoke(self, query):
                profile = build_query_profile(query)
                return keyword_retrieve(self.documents, profile, limit=3)

        retriever = StubRetriever(self.documents)
        cases = [
            RetrievalEvalCase(
                name="kubernetes_case",
                query="What Kubernetes migration work do I have?",
                expected_terms=("kubernetes", "migration"),
            ),
            RetrievalEvalCase(
                name="missing_case",
                query="What Kafka experience do I have?",
                expected_terms=("kafka",),
            ),
        ]

        results = evaluate_retrieval(retriever, cases)
        summary = summarize_retrieval_results(results)

        self.assertTrue(results[0].passed)
        self.assertFalse(results[1].passed)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["failed_cases"], ["missing_case"])


if __name__ == "__main__":
    unittest.main()
