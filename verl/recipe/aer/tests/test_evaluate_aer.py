import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_VERL_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_VERL_ROOT))

from recipe.aer.eval.eval_from_jsonl import main as eval_from_jsonl_main  # noqa: E402
from recipe.aer.eval.eval_from_model import infer_samples_per_prompt  # noqa: E402
from recipe.aer.eval.evaluator import evaluate_records  # noqa: E402
from recipe.aer.eval.metrics.pass_at_k import pass_at_k_unbiased  # noqa: E402
from recipe.aer.eval.metrics.registry import parse_metric_names  # noqa: E402
from recipe.aer.eval.metrics.self_bleu import bleu_score, self_bleu  # noqa: E402
from recipe.aer.eval.metrics.text import tokenize  # noqa: E402
from recipe.aer.eval.train_log import export_tau_plan, parse_train_log  # noqa: E402


class EvaluateAERTest(unittest.TestCase):
    def test_pass_at_k_unbiased(self):
        self.assertAlmostEqual(pass_at_k_unbiased(n_samples=8, n_correct=0, k=4), 0.0)
        self.assertAlmostEqual(pass_at_k_unbiased(n_samples=8, n_correct=8, k=4), 1.0)
        self.assertAlmostEqual(pass_at_k_unbiased(n_samples=8, n_correct=2, k=1), 0.25)

    def test_validation_outputs_summary_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            jsonl_path = tmp_path / "12.jsonl"
            records = [
                {"step": 12, "data_source": "math-ai/math500", "input": "p1", "output": "a b c", "acc": 1.0},
                {"step": 12, "data_source": "math-ai/math500", "input": "p1", "output": "a b d", "acc": 0.0},
                {"step": 12, "data_source": "math-ai/math500", "input": "p2", "output": "x y", "acc": 0.0},
                {"step": 12, "data_source": "math-ai/math500", "input": "p2", "output": "x z", "acc": 0.0},
            ]
            jsonl_path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")

            output_dir = tmp_path / "eval"
            eval_from_jsonl_main(
                [
                    "--input",
                    str(jsonl_path),
                    "--output-dir",
                    str(output_dir),
                    "--ks",
                    "1,2",
                    "--metrics",
                    "pass@k,distinct-2,self-bleu",
                ]
            )

            summary = (output_dir / "validation_summary.csv").read_text(encoding="utf-8")
            self.assertIn("math-ai/math500", summary)
            self.assertIn("pass@1", summary)
            self.assertIn("distinct_2", summary)

    def test_parse_train_log_and_tau_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            log_path = tmp_path / "log.txt"
            lines = []
            for step in range(12, 73, 12):
                reward = 0.5 + step / 1000
                lines.append(
                    f"step:{step} - metric/exploration reward:{reward:.3f} - metric/weight:0.000 - actor/entropy_loss:0.120\n"
                )
            log_path.write_text("".join(lines), encoding="utf-8")

            rows = parse_train_log(log_path)
            self.assertEqual(len(rows), 6)
            self.assertIn("metric/exploration reward", rows[0])

            output_path = tmp_path / "tau.csv"
            export_tau_plan(
                input_path=str(log_path),
                algorithm="token_match",
                output_path=str(output_path),
                precision=3,
            )
            content = output_path.read_text(encoding="utf-8")
            self.assertIn("tau_3", content)
            self.assertIn("token_match", content)

            json_path = tmp_path / "train_metrics.json"
            json_path.write_text(
                json.dumps(
                    [
                        {"step": step, "metric/exploration reward": 0.5 + step / 1000}
                        for step in range(1, 73)
                    ]
                ),
                encoding="utf-8",
            )
            json_output_path = tmp_path / "tau.json"
            row = export_tau_plan(
                input_path=str(json_path),
                algorithm="token_match",
                output_path=str(json_output_path),
                precision=3,
            )
            self.assertEqual(row["tau_1"], 0.513)
            self.assertEqual(row["tau_2"], 0.537)
            self.assertEqual(row["tau_3"], 0.560)

    def test_model_eval_rollout_count_depends_on_pass_metric(self):
        self.assertEqual(infer_samples_per_prompt(parse_metric_names("pass@k,distinct-2"), [1, 2, 4]), 4)
        self.assertEqual(infer_samples_per_prompt(parse_metric_names("distinct-2,self-bleu"), [1, 2, 4]), 1)
        self.assertEqual(infer_samples_per_prompt(parse_metric_names("first@1"), [1, 2, 4]), 1)
        self.assertEqual(infer_samples_per_prompt(parse_metric_names("first@1"), [1, 2, 4], explicit_samples_per_prompt=3), 3)

    def test_self_bleu_matches_candidate_bleu_average(self):
        texts = [
            "We solve x + 1 = 2, so x = 1.",
            "Solving x+1=2 gives x=1.",
            "A different path still gives the final answer 1.",
            "By subtraction, the unknown value is 1.",
        ]
        tokenized = [tokenize(text) for text in texts]
        scores = []
        for idx, candidate in enumerate(tokenized):
            references = [tokens for ref_idx, tokens in enumerate(tokenized) if ref_idx != idx]
            score = bleu_score(candidate, references, max_order=4)
            if score is not None:
                scores.append(score)
        self.assertAlmostEqual(self_bleu(texts, max_order=4), sum(scores) / len(scores))

    def test_validation_groups_by_unique_id_before_prompt_text(self):
        records = [
            {"step": 12, "data_source": "math-ai/mock", "unique_id": "a", "input": "same prompt", "output": "x", "acc": 1.0},
            {"step": 12, "data_source": "math-ai/mock", "unique_id": "b", "input": "same prompt", "output": "y", "acc": 0.0},
        ]
        summary_rows, per_prompt_rows, _, _ = evaluate_records(
            records=records,
            metrics=parse_metric_names("pass@k"),
            ks=[1],
            correct_threshold=0.5,
        )
        dataset_row = next(row for row in summary_rows if row["data_source"] == "math-ai/mock")
        self.assertEqual(dataset_row["n_prompts"], 2)
        self.assertAlmostEqual(dataset_row["pass@1"], 0.5)
        self.assertEqual({row["prompt_id"] for row in per_prompt_rows}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
