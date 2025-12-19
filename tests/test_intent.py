import unittest

from cvrag.rag.intent import detect_query_intent, enhance_query


class IntentTests(unittest.TestCase):
    def test_detect_intent_variants(self):
        self.assertEqual(
            detect_query_intent("Write a cover letter for a DevOps role"),
            "cover_letter",
        )
        self.assertEqual(
            detect_query_intent("Give me STAR examples of leadership"),
            "star_examples",
        )
        self.assertEqual(
            detect_query_intent("Explain the technical details of the migration"),
            "technical_deepdive",
        )
        self.assertEqual(
            detect_query_intent("How should I prepare for interviews?"),
            "interview_prep",
        )
        self.assertEqual(
            detect_query_intent("Create a summary for a role of Cloud Engineer"),
            "role_targeting",
        )
        self.assertEqual(detect_query_intent("What is my AWS experience?"), "general_qa")

    def test_enhance_query(self):
        query = "Write a cover letter for a Site Reliability role"
        enhanced = enhance_query(query)
        self.assertNotEqual(query, enhanced)
        self.assertIn("achievements leadership impact", enhanced)

        general_query = "List my AWS experience"
        enhanced_general = enhance_query(general_query)
        self.assertIn("cloud infrastructure devops", enhanced_general)


if __name__ == "__main__":
    unittest.main()
