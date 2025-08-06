import unittest
import pandas as pd
import re

class TestGSM8KZuluSample(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the dataset once for all tests
        cls.df = pd.read_parquet("QZuluAEnglish.parquet")

    def test_cot_matches_answer_column(self):
        for idx, row in self.df.iterrows():
            cot = row.iloc[1]
            expected_answer = str(row.iloc[2]).strip()

            # Extract answer from CoT (after last '####')
            match = re.search(r"####\s*(-?[\d,]+)\s*$", cot)
            self.assertIsNotNone(match, f"No answer found in CoT at row {idx}")
            cot_answer = match.group(1).replace(",", "").strip()

            # Check match
            self.assertEqual(cot_answer, expected_answer,
                             f"Mismatch at row {idx}: CoT answer '{cot_answer}' != column answer '{expected_answer}'")

if __name__ == "__main__":
    unittest.main()
