# tests/test_pipeline.py

import unittest
from unittest.mock import patch, MagicMock
from MEDRAG.src.main import MedicalDiagnosisPipeline

class TestMedicalDiagnosisPipeline(unittest.TestCase):
    @patch("diagnosis.pipeline.build_knowledge")
    @patch("diagnosis.pipeline.AutoTokenizer.from_pretrained")
    @patch("diagnosis.pipeline.AutoModel.from_pretrained")
    def test_pipeline_run(self, mock_model, mock_tokenizer, mock_build_knowledge):
        # Mock build_knowledge result
        mock_build_knowledge.return_value = {
            "triplets": [("cough", "related_to", "flu")],
            "graph": MagicMock(),
            "level2_clusters": {},
            "level1_clusters": {},
            "embeddings": {"cough": [0.1, 0.2, 0.3]}
        }

        # Mock BioBERT model + tokenizer
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Mocked diagnosis response."

        pipeline = MedicalDiagnosisPipeline(llm=mock_llm)

        result = pipeline.run(["cough", "fever"])

        self.assertEqual(result, "Mocked diagnosis response.")
        mock_llm.generate.assert_called_once()

if __name__ == "__main__":
    unittest.main()
