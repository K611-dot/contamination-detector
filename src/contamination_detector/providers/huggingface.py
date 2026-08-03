"""HuggingFace `transformers` provider for Min-K% and guided-prompting tests.

Requires the `hf` extra (`pip install contamination-detector[hf]`). Kept out
of the core package's hard dependencies so the n-gram detector and report
tooling stay usable without installing torch/transformers.
"""

from __future__ import annotations

from ..min_k_prob import TokenScore

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "HFModelProvider requires the 'hf' extra: pip install contamination-detector[hf]"
    ) from exc


class HFModelProvider:
    """Wraps a local causal LM to produce per-token log-prob stats and completions."""

    def __init__(self, model_name: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def token_scores_for_text(self, text: str) -> list[TokenScore]:
        """Compute per-token logprob, and the vocab mean/std at each position.

        The vocab-level mean/std come from the model's log-softmax output
        over the full vocabulary at each predicted position, as required
        by Min-K%++.
        """
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        if ids.shape[1] < 2:
            return []

        logits = self.model(ids).logits[0]  # (seq_len, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)

        scores: list[TokenScore] = []
        # position i's logits predict token i+1
        for i in range(ids.shape[1] - 1):
            next_token_id = ids[0, i + 1].item()
            dist = log_probs[i]
            scores.append(
                TokenScore(
                    logprob=dist[next_token_id].item(),
                    mean_logprob=dist.mean().item(),
                    std_logprob=dist.std().item(),
                )
            )
        return scores

    @torch.no_grad()
    def complete(self, prefix: str, max_new_tokens: int = 50) -> str:
        ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(self.device)
        output = self.model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output[0, ids.shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
