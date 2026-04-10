import unittest

from cvrag.rag.clarification import build_clarification_response


class ClarificationTests(unittest.TestCase):
    def test_vague_recruiter_query_gets_guidance(self):
        result = build_clarification_response(
            "Summarize the candidate for me",
            "general_qa",
        )

        self.assertIsNotNone(result)
        self.assertTrue(result["needs_clarification"])
        self.assertIn("Which direction should I take for the recruiter?", result["answer"])
        self.assertGreaterEqual(len(result["follow_up_options"]), 4)

    def test_specific_role_targeting_query_skips_guidance(self):
        result = build_clarification_response(
            "Create a summary for a Senior DevOps Engineer role at AWS focusing on Kubernetes and Terraform",
            "role_targeting",
        )

        self.assertIsNone(result)

    def test_vague_interview_prep_query_gets_options(self):
        result = build_clarification_response(
            "Help me prepare for interviews",
            "interview_prep",
        )

        self.assertIsNotNone(result)
        self.assertIn("Which interview angle should I prepare for the recruiter?", result["answer"])
        self.assertIn("Behavioral examples in STAR format", result["follow_up_options"])


if __name__ == "__main__":
    unittest.main()
