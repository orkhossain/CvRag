import unittest

try:
    from langchain_core.documents import Document
except ModuleNotFoundError:  # Allows running tests without langchain installed.
    from dataclasses import dataclass

    @dataclass
    class Document:
        page_content: str
        metadata: dict

from cvrag.rag.formatting import format_docs


class FormattingTests(unittest.TestCase):
    def test_format_docs_order_and_dedupe(self):
        documents = [
            Document(
                page_content="Built a streaming pipeline.",
                metadata={"section": "experience", "company": "Acme"},
            ),
            Document(
                page_content="Built a streaming pipeline.",
                metadata={"section": "experience", "company": "Acme"},
            ),
            Document(
                page_content="Python, AWS, Terraform.",
                metadata={"section": "skills"},
            ),
            Document(
                page_content="BSc Computer Science.",
                metadata={"section": "education"},
            ),
        ]

        result = format_docs(documents)
        lines = result.splitlines()

        self.assertEqual(result.count("Built a streaming pipeline."), 1)
        self.assertTrue(lines[0].startswith("- [EXPERIENCE | Acme]"))
        self.assertIn("[SKILLS]", lines[1])
        self.assertIn("[EDUCATION]", lines[2])


if __name__ == "__main__":
    unittest.main()
