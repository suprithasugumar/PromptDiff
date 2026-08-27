"""Diffing engine combining embedding-drift detection and LLM-as-judge scoring."""

from __future__ import annotations

from promptdiff.embeddings import Embedder, cosine_similarity, get_embedder
from promptdiff.expectations import evaluate_expectations
from promptdiff.judge import LLMJudge
from promptdiff.models import (
    DiffReport,
    ExpectationsCheckResult,
    JudgeVerdict,
    RunOutput,
    TestCase,
    TestCaseDiffResult,
    TestSuite,
)
from promptdiff.providers import get_provider


class DiffEngine:
    """Computes semantic drift and coordinates LLM-as-judge evaluation against baseline."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        judge: LLMJudge | None = None,
        similarity_threshold: float = 0.88,
    ) -> None:
        self.embedder = embedder or get_embedder()
        if judge is None:
            provider = get_provider("gemini")
            self.judge = LLMJudge(provider=provider)
        else:
            self.judge = judge
        self.similarity_threshold = similarity_threshold

    def diff(
        self,
        baseline_run: RunOutput,
        current_run: RunOutput,
        suite: TestSuite,
        dry_run: bool = False,
        progress_callback=None,
    ) -> DiffReport:
        """Diff the current run against the baseline run."""
        baseline_map = {r.test_case_id: r for r in baseline_run.results}
        case_map: dict[str, TestCase] = {tc.id: tc for tc in suite.test_cases}

        # Step 1: Collect output pairs for batch embedding
        pairs_to_embed: list[tuple[str, str, str]] = []  # (case_id, base_out, curr_out)
        for curr_res in current_run.results:
            base_res = baseline_map.get(curr_res.test_case_id)
            if base_res and base_res.output and curr_res.output and not curr_res.error:
                pairs_to_embed.append(
                    (curr_res.test_case_id, base_res.output, curr_res.output)
                )

        # Batch embed outputs
        similarity_map: dict[str, float] = {}
        if pairs_to_embed:
            if dry_run:
                # Use MockEmbedder in dry-run mode
                mock_embedder = get_embedder(mock=True)
                base_texts = [p[1] for p in pairs_to_embed]
                curr_texts = [p[2] for p in pairs_to_embed]
                base_vecs = mock_embedder.embed(base_texts)
                curr_vecs = mock_embedder.embed(curr_texts)
                for (cid, _, _), v_base, v_curr in zip(pairs_to_embed, base_vecs, curr_vecs):
                    similarity_map[cid] = round(cosine_similarity(v_base, v_curr), 3)
            else:
                base_texts = [p[1] for p in pairs_to_embed]
                curr_texts = [p[2] for p in pairs_to_embed]
                all_texts = base_texts + curr_texts
                all_vecs = self.embedder.embed(all_texts)
                n = len(base_texts)
                base_vecs = all_vecs[:n]
                curr_vecs = all_vecs[n:]
                for (cid, _, _), v_base, v_curr in zip(pairs_to_embed, base_vecs, curr_vecs):
                    similarity_map[cid] = round(cosine_similarity(v_base, v_curr), 3)

        # Step 2: Evaluate expectations and judge flagged cases
        diff_results: list[TestCaseDiffResult] = []
        passed_count = 0
        regressed_count = 0
        improved_count = 0
        error_count = 0
        judge_calls_count = 0

        for curr_res in current_run.results:
            cid = curr_res.test_case_id
            base_res = baseline_map.get(cid)
            case = case_map.get(cid, TestCase(id=cid, input=curr_res.input))
            input_text = curr_res.input

            # Compute latency and token deltas
            base_latency = base_res.latency_ms if base_res else 0.0
            base_tokens = (
                (base_res.prompt_tokens + base_res.completion_tokens) if base_res else 0
            )
            curr_tokens = curr_res.prompt_tokens + curr_res.completion_tokens
            latency_delta = round(curr_res.latency_ms - base_latency, 2)
            token_delta = curr_tokens - base_tokens

            if curr_res.error:
                error_count += 1
                diff_results.append(
                    TestCaseDiffResult(
                        test_case_id=cid,
                        input=input_text,
                        status="error",
                        baseline_output=base_res.output if base_res else None,
                        new_output=None,
                        error=curr_res.error,
                        latency_delta_ms=latency_delta,
                        token_delta=token_delta,
                    )
                )
                continue

            # Check programmatic expectations
            exp_result = evaluate_expectations(curr_res.output, case.expectations)
            similarity = similarity_map.get(cid)

            # Determine whether to flag for LLM Judge
            flag_reasons: list[str] = []
            if not exp_result.passed:
                flag_reasons.extend(exp_result.failures)

            if similarity is not None and similarity < self.similarity_threshold:
                flag_reasons.append(
                    f"Embedding drift detected: similarity {similarity:.3f} < threshold {self.similarity_threshold:.2f}"
                )

            flagged = len(flag_reasons) > 0
            verdict: JudgeVerdict | None = None

            if flagged and base_res and base_res.output and curr_res.output:
                judge_calls_count += 1
                if dry_run:
                    verdict = JudgeVerdict(
                        reasoning="[DRY RUN] Flagged case simulated in dry run.",
                        verdict="worse" if not exp_result.passed else "equivalent",
                        category="simulated_drift" if similarity and similarity < self.similarity_threshold else "expectation_failure",
                        confidence=1.0,
                    )
                else:
                    try:
                        verdict = self.judge.evaluate(
                            case=case,
                            baseline_output=base_res.output,
                            new_output=curr_res.output,
                            system_prompt=suite.target.system_prompt,
                        )
                    except Exception as err:
                        verdict = JudgeVerdict(
                            reasoning=f"Judge evaluation failed: {err}",
                            verdict="worse" if not exp_result.passed else "equivalent",
                            category="judge_error",
                            confidence=0.0,
                        )

            # Determine status based on expectations and verdict
            if verdict:
                if verdict.verdict == "worse":
                    status = "regressed"
                    regressed_count += 1
                elif verdict.verdict == "better":
                    status = "improved"
                    improved_count += 1
                else:
                    if not exp_result.passed:
                        status = "regressed"
                        regressed_count += 1
                    else:
                        status = "pass"
                        passed_count += 1
            else:
                if not exp_result.passed:
                    status = "regressed"
                    regressed_count += 1
                else:
                    status = "pass"
                    passed_count += 1

            case_diff = TestCaseDiffResult(
                test_case_id=cid,
                input=input_text,
                status=status,
                similarity_score=similarity,
                baseline_output=base_res.output if base_res else None,
                new_output=curr_res.output,
                expectations_result=exp_result,
                flagged_for_judge=flagged,
                flag_reasons=flag_reasons,
                judge_verdict=verdict,
                latency_delta_ms=latency_delta,
                token_delta=token_delta,
            )
            diff_results.append(case_diff)
            if progress_callback:
                progress_callback(case_diff)

        return DiffReport(
            suite_name=suite.name,
            baseline_run_id=baseline_run.run_id,
            baseline_timestamp=baseline_run.timestamp,
            new_run_id=current_run.run_id,
            new_timestamp=current_run.timestamp,
            target=current_run.target,
            total_cases=len(diff_results),
            passed_cases=passed_count,
            regressed_cases=regressed_count,
            improved_cases=improved_count,
            error_cases=error_count,
            judge_calls_count=judge_calls_count,
            case_diffs=diff_results,
        )
