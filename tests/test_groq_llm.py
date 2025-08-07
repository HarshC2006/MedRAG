import unittest
from unittest.mock import patch, MagicMock
from llm.groq_llm import GroqLLM

class TestGroqLLM(unittest.TestCase):

    @patch("llm.groq_llm.Groq")
    def test_generate_response(self, mock_groq_class):
        # Mock the Groq model and its completion behavior
        mock_llm_instance = MagicMock()
        mock_llm_instance.complete.return_value.text = "mocked response"
        mock_groq_class.return_value = mock_llm_instance

        llm = GroqLLM()
        response = llm.generate("What is the capital of France?")

        self.assertEqual(response, "mocked response")
        mock_llm_instance.complete.assert_called_once()

if __name__ == "__main__":
    unittest.main()