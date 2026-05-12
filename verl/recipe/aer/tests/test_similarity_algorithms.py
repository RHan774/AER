import sys
import types
import unittest
from pathlib import Path

import torch

PROJECT_VERL_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_VERL_ROOT))
if "verl" not in sys.modules:
    verl_stub = types.ModuleType("verl")
    verl_stub.DataProto = object
    sys.modules["verl"] = verl_stub

from recipe.aer.src.similarity import get_similarity_computer, list_available_algorithms  # noqa: E402
from recipe.aer.src.reward_manager import _normalize_metric_algorithms  # noqa: E402


class FakeTokenizer:
    def __init__(self, mapping, join_with_space=False):
        self.mapping = mapping
        self.joiner = " " if join_with_space else ""

    def decode(self, token_ids, skip_special_tokens=True):
        return self.joiner.join(self.mapping[int(token_id)] for token_id in token_ids if int(token_id) != 0)


class FakeItem:
    def __init__(self, batch, non_tensor_batch):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch


class FakeDataProto:
    def __init__(self, responses, uids, prompt_len=2):
        batch_size = len(responses)
        max_response_len = max(len(response) for response in responses)

        prompt_tensor = torch.full((batch_size, prompt_len), 101, dtype=torch.long)
        response_tensor = torch.zeros((batch_size, max_response_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, prompt_len + max_response_len), dtype=torch.long)

        for row, response in enumerate(responses):
            response_tensor[row, : len(response)] = torch.tensor(response, dtype=torch.long)
            attention_mask[row, :prompt_len] = 1
            attention_mask[row, prompt_len : prompt_len + len(response)] = 1

        self.batch = {
            "prompts": prompt_tensor,
            "responses": response_tensor,
            "attention_mask": attention_mask,
        }
        self.non_tensor_batch = {"uid": list(uids)}

    def __len__(self):
        return self.batch["responses"].shape[0]

    def __getitem__(self, idx):
        batch = {key: value[idx] for key, value in self.batch.items()}
        non_tensor_batch = {key: value[idx] for key, value in self.non_tensor_batch.items()}
        return FakeItem(batch=batch, non_tensor_batch=non_tensor_batch)


class SimilarityAlgorithmTest(unittest.TestCase):
    def test_token_match_matches_baseline_formula(self):
        data = FakeDataProto([[1, 2, 3], [1, 2, 4], [5, 6]], ["g1", "g1", "g2"])
        computer = get_similarity_computer("token_match")
        matrix = computer.compute(data)

        self.assertAlmostEqual(matrix[0, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(matrix[1, 1].item(), 1.0, places=6)
        self.assertAlmostEqual(matrix[0, 1].item(), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(matrix[1, 0].item(), 2.0 / 3.0, places=6)
        self.assertAlmostEqual(matrix[0, 2].item(), 0.0, places=6)
        self.assertAlmostEqual(matrix[2, 2].item(), 1.0, places=6)

    def test_ngram_overlap_uses_jaccard_within_group_only(self):
        data = FakeDataProto([[1, 2, 3], [1, 2, 4], [5, 6]], ["g1", "g1", "g2"])
        computer = get_similarity_computer("ngram_overlap", n=2)
        matrix = computer.compute(data)

        self.assertAlmostEqual(matrix[0, 1].item(), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(matrix[0, 2].item(), 0.0, places=6)
        self.assertAlmostEqual(matrix[2, 2].item(), 1.0, places=6)

    def test_ngram_overlap_counts_repeated_ngrams(self):
        data = FakeDataProto([[1, 1, 1, 1], [1, 1, 1]], ["g1", "g1"])
        computer = get_similarity_computer("ngram_overlap", n=2)
        matrix = computer.compute(data)

        self.assertAlmostEqual(matrix[0, 1].item(), 2.0 / 3.0, places=6)

    def test_ngram_overlap_returns_zero_when_sequence_is_shorter_than_n(self):
        data = FakeDataProto([[1, 2], [1, 2]], ["g1", "g1"])
        computer = get_similarity_computer("ngram_overlap", n=3)
        matrix = computer.compute(data)

        self.assertAlmostEqual(matrix[0, 0].item(), 0.0, places=6)
        self.assertAlmostEqual(matrix[0, 1].item(), 0.0, places=6)
        self.assertAlmostEqual(matrix[1, 1].item(), 0.0, places=6)

    def test_char_ngram_jaccard_matches_expected_value(self):
        tokenizer = FakeTokenizer({1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "x", 7: "y"})
        data = FakeDataProto([[1, 2, 3, 4], [1, 2, 5, 4], [6, 7]], ["g1", "g1", "g2"])
        computer = get_similarity_computer("char_ngram", n=2, metric="jaccard")
        matrix = computer.compute(data, tokenizer=tokenizer)

        self.assertAlmostEqual(matrix[0, 1].item(), 1.0 / 5.0, places=6)
        self.assertAlmostEqual(matrix[0, 2].item(), 0.0, places=6)
        self.assertAlmostEqual(matrix[1, 1].item(), 1.0, places=6)

    def test_levenshtein_similarity_matches_normalized_definition(self):
        mapping = {1: "k", 2: "i", 3: "t", 4: "e", 5: "n", 6: "s", 7: "a", 8: "b", 9: "c"}
        tokenizer = FakeTokenizer(mapping)
        data = FakeDataProto([[1, 2, 3, 3, 4, 5], [6, 2, 3, 3, 4, 5], [7, 8, 9]], ["g1", "g1", "g2"])
        computer = get_similarity_computer("levenshtein", normalize_method="max")
        matrix = computer.compute(data, tokenizer=tokenizer)

        self.assertAlmostEqual(matrix[0, 1].item(), 5.0 / 6.0, places=6)
        self.assertAlmostEqual(matrix[0, 2].item(), 0.0, places=6)
        self.assertAlmostEqual(matrix[1, 1].item(), 1.0, places=6)

    def test_levenshtein_normalizes_unicode_before_comparison(self):
        tokenizer = FakeTokenizer({1: "é", 2: "e\u0301"})
        data = FakeDataProto([[1], [2]], ["g1", "g1"])
        computer = get_similarity_computer("levenshtein", normalize_method="max")
        matrix = computer.compute(data, tokenizer=tokenizer)

        self.assertAlmostEqual(matrix[0, 1].item(), 1.0, places=6)

    def test_tfidf_cosine_returns_groupwise_cosine_matrix(self):
        tokenizer = FakeTokenizer(
            {1: "red", 2: "blue", 3: "green", 4: "cat", 5: "dog"},
            join_with_space=True,
        )
        data = FakeDataProto([[1, 2], [1, 3], [4, 5]], ["g1", "g1", "g2"])
        computer = get_similarity_computer("tfidf_cosine", ngram_range=(1, 1), max_features=None)
        matrix = computer.compute(data, tokenizer=tokenizer)

        self.assertGreater(matrix[0, 1].item(), 0.0)
        self.assertLess(matrix[0, 1].item(), 1.0)
        self.assertAlmostEqual(matrix[0, 2].item(), 0.0, places=6)
        self.assertAlmostEqual(matrix[2, 2].item(), 1.0, places=6)

    def test_compression_similarity_is_symmetric_and_grouped(self):
        tokenizer = FakeTokenizer({1: "a", 2: "b", 3: "c", 4: "x", 5: "y", 6: "z"})
        data = FakeDataProto([[1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], [4, 5, 6, 4, 5, 6]], ["g1", "g1", "g2"])
        computer = get_similarity_computer("compression_ratio")
        matrix = computer.compute(data, tokenizer=tokenizer)

        self.assertAlmostEqual(matrix[0, 1].item(), matrix[1, 0].item(), places=6)
        self.assertGreaterEqual(matrix[0, 1].item(), 0.99)
        self.assertAlmostEqual(matrix[0, 2].item(), 0.0, places=6)
        self.assertAlmostEqual(matrix[1, 1].item(), 1.0, places=6)

    def test_rouge_l_matches_lcs_f_score(self):
        tokenizer = FakeTokenizer({1: "a", 2: "b", 3: "c"}, join_with_space=True)
        data = FakeDataProto([[1, 2, 3], [1, 3], [2]], ["g1", "g1", "g2"])
        computer = get_similarity_computer("rouge_l", beta=1.0)
        matrix = computer.compute(data, tokenizer=tokenizer)

        self.assertAlmostEqual(matrix[0, 1].item(), 0.8, places=6)
        self.assertAlmostEqual(matrix[1, 0].item(), 0.8, places=6)
        self.assertAlmostEqual(matrix[0, 2].item(), 0.0, places=6)

    def test_factory_can_build_semantic_embedding(self):
        computer = get_similarity_computer("semantic_embedding", model_name="all-MiniLM-L6-v2")
        self.assertEqual(computer.model_name, "all-MiniLM-L6-v2")

    def test_semantic_embedding_accepts_dedicated_cuda_devices(self):
        computer = get_similarity_computer(
            "semantic_embedding",
            model_name="unused",
            device="cuda",
            cuda_visible_devices="4,5,6,7",
            num_processes=4,
        )

        self.assertEqual(computer.cuda_visible_devices, ["4", "5", "6", "7"])
        self.assertEqual(computer.cuda_visible_devices_key, "4,5,6,7")

    def test_semantic_embedding_normalizes_to_closed_interval_and_keeps_groups_separate(self):
        tokenizer = FakeTokenizer({1: "same", 2: "opposite", 3: "cross"}, join_with_space=True)
        data = FakeDataProto([[1], [2], [3]], ["g1", "g1", "g2"])
        computer = get_similarity_computer("semantic_embedding", model_name="unused")
        computer._encode_texts = lambda texts: torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])

        matrix = computer.compute(data, tokenizer=tokenizer)

        self.assertAlmostEqual(matrix[0, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(matrix[0, 1].item(), 0.0, places=6)
        self.assertAlmostEqual(matrix[0, 2].item(), 0.0, places=6)
        self.assertAlmostEqual(matrix[2, 2].item(), 1.0, places=6)

    def test_semantic_embedding_encodes_duplicate_texts_once(self):
        tokenizer = FakeTokenizer({1: "same", 2: "other"}, join_with_space=True)
        data = FakeDataProto([[1], [1], [2]], ["g1", "g1", "g1"])
        computer = get_similarity_computer("semantic_embedding", model_name="unused")
        encoded_texts = []

        def fake_encode_batch(texts):
            encoded_texts.extend(texts)
            return torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        computer._encode_batch = fake_encode_batch
        matrix = computer.compute(data, tokenizer=tokenizer)

        self.assertEqual(encoded_texts, ["same", "other"])
        self.assertAlmostEqual(matrix[0, 1].item(), 1.0, places=6)
        self.assertAlmostEqual(matrix[0, 2].item(), 0.5, places=6)

    def test_simhash_is_symmetric_and_grouped(self):
        data = FakeDataProto([[1, 2, 3, 4], [1, 2, 3, 4], [9, 8, 7, 6]], ["g1", "g1", "g2"])
        computer = get_similarity_computer("simhash", n=2, hash_bits=32)
        matrix = computer.compute(data)

        self.assertAlmostEqual(matrix[0, 1].item(), 1.0, places=6)
        self.assertAlmostEqual(matrix[1, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(matrix[0, 2].item(), 0.0, places=6)
        self.assertAlmostEqual(matrix[2, 2].item(), 1.0, places=6)

    def test_available_algorithms_include_all_algorithms(self):
        algorithms = list_available_algorithms()
        self.assertIn("token_match", algorithms)
        self.assertIn("ngram_overlap", algorithms)
        self.assertIn("char_ngram", algorithms)
        self.assertIn("levenshtein", algorithms)
        self.assertIn("tfidf_cosine", algorithms)
        self.assertIn("semantic_embedding", algorithms)
        self.assertIn("simhash", algorithms)
        self.assertIn("compression_ratio", algorithms)
        self.assertIn("rouge_l", algorithms)

    def test_metric_algorithm_list_accepts_hydra_list_string(self):
        algorithms = _normalize_metric_algorithms("[token_match,ngram_overlap,rouge_l]")

        self.assertEqual(algorithms, ["token_match", "ngram_overlap", "rouge_l"])

    def test_metric_algorithm_list_expands_all(self):
        algorithms = _normalize_metric_algorithms(["all"])

        self.assertEqual(algorithms, list_available_algorithms())


if __name__ == "__main__":
    unittest.main()
