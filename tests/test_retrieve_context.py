import unittest

try:
    from langchain_core.documents import Document
except ModuleNotFoundError:  # Allows running tests without langchain installed.
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict

from cvrag.rag.graph import retrieve_context_node
from cvrag.rag.retriever import set_retriever


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.documents


class RetrieveContextTests(unittest.TestCase):
    def test_retrieve_context_dedupes_and_enhances(self):
        documents = [
            Document(page_content="Reduced latency by 30%.", metadata={"section": "skills"}),
            Document(page_content="Reduced latency by 30%.", metadata={"section": "skills"}),
            Document(page_content="Led migration to Kubernetes.", metadata={"section": "experience"}),
        ]
        fake = FakeRetriever(documents)
        set_retriever(fake)

        state = {
            "query": "Explain the technical details of the migration project",
            "intent": "technical_deepdive",
        }
        result = retrieve_context_node(state)

        self.assertTrue(result.get("query_enhanced"))
        self.assertEqual(result["context"].count("Reduced latency by 30%"), 1)
        self.assertEqual(result["context_used"], len(result["context"].splitlines()))
        self.assertGreaterEqual(len(fake.queries), 2)

        set_retriever(None)


if __name__ == "__main__":
    unittest.main()
