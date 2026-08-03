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

    def __init__(self, model_name: str, device: str | None = None, dtype=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    @property
    def max_length(self) -> int:
        limit = getattr(self.model.config, "max_position_embeddings", None)
        # Some configs report a sentinel-large value; fall back to a sane cap.
        if not limit or limit > 1_000_000:
            return 2048
        return int(limit)

    @torch.no_grad()
    def token_scores_for_text(self, text: str) -> list[TokenScore]:
        """Compute per-token logprob plus the Min-K%++ calibration statistics.

        mu and sigma are expectations under the model's own next-token
        distribution (mu is the negative entropy), matching the Min-K%++
        definition. Everything is computed as vectorised tensor ops over the
        whole sequence — the per-token Python loop this replaced forced a
        device sync on every single token.
        """
        ids = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).input_ids.to(self.device)
        if ids.shape[1] < 2:
            return []

        # Position i predicts token i+1, so drop the final position's logits.
        logits = self.model(ids).logits[0, :-1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # mu = E_{z~p}[log p(z)] = -entropy; sigma likewise weighted by p.
        mu = (probs * log_probs).sum(dim=-1)
        variance = (probs * (log_probs - mu.unsqueeze(-1)) ** 2).sum(dim=-1)
        sigma = variance.clamp_min(0).sqrt()

        targets = ids[0, 1:]
        observed = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # One host transfer for the whole sequence rather than 3 per token.
        observed_list = observed.tolist()
        mu_list = mu.tolist()
        sigma_list = sigma.tolist()

        return [
            TokenScore(logprob=o, expected_logprob=m, logprob_std=s)
            for o, m, s in zip(observed_list, mu_list, sigma_list)
        ]

    @torch.no_grad()
    def complete(self, prefix: str, max_new_tokens: int = 50) -> str:
        ids = self.tokenizer(
            prefix,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).input_ids.to(self.device)
        output = self.model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        generated_ids = output[0, ids.shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
