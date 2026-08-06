"""MoEModelRunner: batched HF forward with paged KV as attention source of truth."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import torch

from .kernel_backend import PagedAttnContext, create_kernel_backend
from .kv_cache import PastKV, RadixCache
from .models import assert_moe_supported, is_fixture_model, register_verified
from .sampling import make_generator, sample_logits
from .scheduler import Sequence


class MoERouter:
    """Placeholder router; real MoE routing is inside the HF model."""

    def route(self, input_ids: List[int]) -> List[int]:
        return [0] * len(input_ids)


class ModelRunner:
    def __init__(
        self,
        model_name: str = "stub",
        device: str = "cpu",
        max_batch: int = 4,
        allow_stub: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.max_batch = max_batch
        self.allow_stub = allow_stub

        self.model = None
        self.tokenizer = None
        self._is_real = False
        self.vocab_size = 32000
        self.num_layers = 4
        self.num_kv_heads = 4
        self.head_dim = 64
        self.eos_token_id: Optional[int] = 2
        # Observability for tests: size of the last tensor forward (batch dim)
        self.last_model_forward_size = 0
        self.model_forward_count = 0
        self.paged_rebuild_count = 0
        self.use_paged_as_source = True
        self._v4_hybrid = False
        self._v4_prefix_cache = None
        self._tp_plan = None
        self._kv_layout = None
        self._swa_layout = None
        # Must match model compute dtype; fp16 pages + bf16 activations promote to fp32 in SDPA.
        self.torch_dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.kernel_backend = create_kernel_backend(self.device)

        if model_name == "stub":
            if not allow_stub:
                raise ValueError(
                    "model='stub' is only allowed with allow_stub=True "
                    "(explicit demo/test mode). Production must load a real MoE model."
                )
            self._init_tiny_stub_model()
        else:
            self._load_real(model_name)

    def _load_real(self, model_name: str) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        assert_moe_supported(model_name)

        if is_fixture_model(model_name):
            path = model_name.split(":", 1)[1]
            load_id = path
        else:
            load_id = model_name

        print(f"[sglang-lite] Loading MoE model: {load_id}")
        config = AutoConfig.from_pretrained(load_id, trust_remote_code=True)
        model_type = getattr(config, "model_type", None)
        fam = assert_moe_supported(model_name, model_type)

        self.tokenizer = AutoTokenizer.from_pretrained(
            load_id, trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if fam.name == "deepseek_v4" or (model_type or "").lower() == "deepseek_v4":
            self._load_deepseek_v4_hybrid(model_name, load_id, config)
            return

        dtype = self.torch_dtype
        load_kwargs = {
            "dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        use_device_map = False
        if self.device != "cpu":
            try:
                import accelerate  # noqa: F401

                load_kwargs["device_map"] = "auto"
                use_device_map = True
            except ImportError:
                use_device_map = False
        self.model = AutoModelForCausalLM.from_pretrained(load_id, **load_kwargs)
        if not use_device_map:
            self.model = self.model.to(self.device)
        self.model.eval()
        self._is_real = True
        self.vocab_size = int(self.model.config.vocab_size)
        self.num_layers = int(self.model.config.num_hidden_layers)
        self.num_kv_heads = int(
            getattr(
                self.model.config,
                "num_key_value_heads",
                self.model.config.num_attention_heads,
            )
        )
        self.head_dim = int(
            getattr(self.model.config, "head_dim", None)
            or (self.model.config.hidden_size // self.model.config.num_attention_heads)
        )
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None) or getattr(
            self.model.config, "eos_token_id", 2
        )

        print(f"[sglang-lite] Kernel backend: {self.kernel_backend.name}")
        if self.kernel_backend.supports_paged_attention:
            sm_scale = self.head_dim**-0.5
            n_heads = int(self.model.config.num_attention_heads)
            n = self.kernel_backend.attach_to_model(
                self.model, num_qo_heads=n_heads, head_dim=self.head_dim, sm_scale=sm_scale
            )
            print(f"[sglang-lite] Paged attention hooks attached: {n} modules")

        register_verified(model_name)
        print(f"[sglang-lite] MoE model '{model_name}' ready on {self.device}")

    def _load_deepseek_v4_hybrid(self, model_name: str, load_id: str, config) -> None:
        """Hybrid: official inference Transformer + TP shard from convert.py."""
        import os

        from .kv_cache import KvLayout
        from .model_loader import load_v4_hybrid_model, resolve_v4_paths

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        paths = resolve_v4_paths(hf_ckpt=load_id)
        if paths.converted_ckpt is None:
            raise RuntimeError(
                "deepseek_v4 Hybrid load requires converted MP shards. "
                "Run scripts/v4_official_smoke.sh then set "
                "SGLANG_LITE_DSV4_CONVERTED=/tmp/ds-v4-mp8 (and torchrun for TP)."
            )
        if world_size < 2:
            raise RuntimeError(
                "deepseek_v4 requires tensor parallel (WORLD_SIZE>=2, typically 8). "
                "Use: torchrun --nproc-per-node=8 scripts/v4_lite_engine_gen.py "
                "or scripts/v4_lite_short_gen.py"
            )
        # load_v4_hybrid_model inits NCCL before Transformer; device from LOCAL_RANK.
        model, cfg, plan = load_v4_hybrid_model(
            paths, max_batch_size=self.max_batch
        )
        self.model = model
        self.device = plan.local_device
        self.torch_dtype = torch.bfloat16
        self._is_real = True
        self._v4_hybrid = True
        from .v4_prefix_cache import V4PrefixCache

        self._v4_prefix_cache = V4PrefixCache()
        self._tp_plan = plan
        self.vocab_size = int(cfg.get("vocab_size", 129280))
        self.num_layers = int(cfg.get("n_layers") or cfg.get("num_hidden_layers", 43))
        self.num_kv_heads = int(cfg.get("num_key_value_heads", 1))
        self.head_dim = int(cfg.get("head_dim", 512))
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None) or int(
            cfg.get("eos_token_id", 1)
        )
        # V4 uses MLA / sparse pools — standard MHA hooks do not apply.
        self.use_paged_as_source = False
        self._kv_layout = KvLayout.mla_compressed(
            ckv_dim=int(cfg.get("head_dim", 512)),
            kpe_dim=int(cfg.get("rope_head_dim") or cfg.get("qk_rope_head_dim", 64)),
        )
        self._swa_layout = KvLayout.dsv4_packed(584)
        kb = self.kernel_backend
        # Match official generate.py determinism for greedy numerical checks.
        torch.manual_seed(33377335)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(33377335)
        # Refresh capabilities after Hybrid import path may have put FI≥0.6.16
        # on sys.path (SGLANG_LITE_FI_PREFIX). Then arm sparse MLA hook.
        disable_fi = os.environ.get("SGLANG_LITE_V4_DISABLE_FI_SPARSE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        try:
            from .capability import probe_kernel_capabilities
            from .kernel_backend import FlashInferBackend
            from .v4_sparse_mla import attach_v4_sparse_mla

            from .capability import SparseMlaBackend

            caps = probe_kernel_capabilities(self.device)
            if isinstance(kb, FlashInferBackend):
                kb.capabilities = caps
                kb.supports_sparse_mla = caps.sparse_mla_backend in (
                    SparseMlaBackend.FLASHINFER_SPARSE_SM120,
                    SparseMlaBackend.FLASHINFER_SPARSE_SM100,
                )
            win = int(cfg.get("window_size", 128))
            if disable_fi:
                print(
                    "[sglang-lite] v4 sparse MLA hook disabled "
                    "(SGLANG_LITE_V4_DISABLE_FI_SPARSE); official sparse_attn only"
                )
                armed = False
            else:
                armed = attach_v4_sparse_mla(kb, window_size=win)
            print(
                f"[sglang-lite] v4 sparse MLA hook armed={armed} "
                f"backend={kb.sparse_mla_backend.value}"
            )
        except Exception as e:
            print(f"[sglang-lite] v4 sparse MLA hook skipped: {e}")
        print(
            f"[sglang-lite] deepseek_v4 Hybrid rank={plan.rank}/{plan.world_size} "
            f"arch={kb.arch_family.value} sparse_mla={kb.sparse_mla_backend.value} "
            f"moe_gemm={kb.moe_gemm_backend.value}"
        )
        register_verified(model_name)
        print(f"[sglang-lite] MoE model '{model_name}' Hybrid ready on {plan.local_device}")

    def _init_tiny_stub_model(self):
        import torch.nn as nn

        class TinyLM(nn.Module):
            def __init__(self, vocab=32000, dim=256, layers=4, heads=4):
                super().__init__()
                self.embed = nn.Embedding(vocab, dim)
                self.layers = nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True)
                        for _ in range(layers)
                    ]
                )
                self.ln = nn.LayerNorm(dim)
                self.head = nn.Linear(dim, vocab, bias=False)
                self.vocab_size = vocab
                self.config = type(
                    "C",
                    (),
                    {
                        "vocab_size": vocab,
                        "num_hidden_layers": layers,
                        "num_attention_heads": heads,
                        "num_key_value_heads": heads,
                        "hidden_size": dim,
                        "eos_token_id": 2,
                    },
                )()

            def forward(
                self, input_ids, past_key_values=None, use_cache=False, attention_mask=None, **kwargs
            ):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer(x)
                x = self.ln(x)
                logits = self.head(x)
                dummy_past = None
                if use_cache:
                    dummy_past = [
                        (x[:, -1:, :].clone(), x[:, -1:, :].clone())
                        for _ in range(len(self.layers))
                    ]
                return type(
                    "Obj",
                    (),
                    {"logits": logits, "past_key_values": dummy_past},
                )()

        self.model = TinyLM(vocab=self.vocab_size).to(self.device)
        self.model.eval()
        self._is_real = False
        self.num_layers = 4
        self.num_kv_heads = 4
        self.head_dim = 64
        self.eos_token_id = 2
        self.use_paged_as_source = False  # stub past shapes are not real KV
        print("[sglang-lite] Using explicit tiny stub model (allow_stub=True).")

    def tokenize(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return [hash(c) % self.vocab_size for c in text[:128]]

    def apply_chat_template(self, messages: list) -> List[int]:
        if self._v4_hybrid:
            ids = self._v4_encode_messages(messages)
            if ids is not None:
                return ids
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                ids = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
                return list(ids)
            except Exception:
                pass
        parts = []
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content") or ""
            else:
                role = getattr(m, "role", "user")
                content = getattr(m, "content", "") or ""
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return self.tokenize("\n".join(parts))

    def _v4_encode_messages(self, messages: list) -> Optional[List[int]]:
        """Prefer official encoding_dsv4 when HF tree is available."""
        import sys

        try:
            from .model_loader import resolve_v4_paths

            paths = resolve_v4_paths()
        except Exception:
            return None
        encoding_dir = paths.hf_ckpt / "encoding"
        infer_dir = paths.hf_ckpt / "inference"
        if not encoding_dir.is_dir():
            return None
        for p in (str(encoding_dir), str(infer_dir)):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            from encoding_dsv4 import encode_messages

            norm = []
            for m in messages:
                if isinstance(m, dict):
                    norm.append(
                        {
                            "role": m.get("role", "user"),
                            "content": m.get("content") or "",
                        }
                    )
                else:
                    norm.append(
                        {
                            "role": getattr(m, "role", "user"),
                            "content": getattr(m, "content", "") or "",
                        }
                    )
            rendered = encode_messages(norm, thinking_mode="chat")
            if self.tokenizer is None:
                return None
            return list(self.tokenizer.encode(rendered))
        except Exception:
            return None

    def detokenize(self, token_ids: List[int]) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return "".join([chr(97 + (t % 26)) for t in token_ids[:20]])

    def detokenize_delta(self, token_ids: List[int], prev_text: str = "") -> str:
        """Incremental decode; never re-emit the whole string on tokenizer churn.

        - Prefers ``full[len(prev):]`` when ``full`` extends ``prev``.
        - If ``full`` is a prefix of ``prev`` (incomplete multi-byte piece), wait.
        - Otherwise emit only the suffix after the longest common prefix.
        - If there is no common prefix and ``prev`` is non-empty, emit nothing
          (avoids dumping a rewritten full sentence / ``�`` storms to SSE).
        """
        if not token_ids:
            return ""
        full = self.detokenize(token_ids)
        if full.startswith(prev_text):
            delta = full[len(prev_text) :]
        elif prev_text.startswith(full):
            # Incomplete UTF-8 / multi-token glyph — hold until decode grows.
            return ""
        else:
            n = 0
            limit = min(len(prev_text), len(full))
            while n < limit and prev_text[n] == full[n]:
                n += 1
            if n == 0 and prev_text:
                return ""
            delta = full[n:]
        # Lone U+FFFD is an incomplete decode artifact — do not stream it.
        if delta == "\ufffd":
            return ""
        return delta

    @torch.no_grad()
    def run_batch(
        self,
        batch: List[Sequence],
        radix: RadixCache,
        is_prefill: List[bool],
    ) -> List[Optional[int]]:
        """Execute one scheduler step.

        Prefill/decode groups that share the same cached length are executed with a
        single batched model forward (tensor batch dim > 1 when possible).
        """
        if not batch:
            return []

        results: List[Optional[int]] = [None] * len(batch)
        self.last_model_forward_size = 0

        prefill_idxs: List[int] = []
        decode_idxs: List[int] = []

        for i, seq in enumerate(batch):
            if seq.finished:
                continue
            # Exact prefix hit: first token from stored logits (no forward)
            if not seq.output_ids and getattr(seq, "last_logits", None) is not None:
                if seq.cached_len >= len(seq.input_ids) or (
                    not is_prefill[i] and seq.cached_len == len(seq.input_ids)
                ):
                    if self._v4_hybrid:
                        self._v4_ensure_restored(seq, batch_slot=0)
                    results[i] = self._sample(seq.last_logits.to(device=self.device), seq)
                    seq.cached_len = len(seq.input_ids)
                    continue
            if is_prefill[i] and seq.cached_len < len(seq.input_ids):
                prefill_idxs.append(i)
            else:
                decode_idxs.append(i)

        if prefill_idxs:
            self._run_prefill_groups(batch, prefill_idxs, radix, results)
        if decode_idxs:
            self._run_decode_groups(batch, decode_idxs, radix, results)
        return results

    def _run_prefill_groups(
        self,
        batch: List[Sequence],
        idxs: List[int],
        radix: RadixCache,
        results: List[Optional[int]],
    ) -> None:
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in idxs:
            groups[batch[i].cached_len].append(i)
        for _cached_len, group in groups.items():
            self._batch_prefill([batch[i] for i in group], group, radix, results)

    def _run_decode_groups(
        self,
        batch: List[Sequence],
        idxs: List[int],
        radix: RadixCache,
        results: List[Optional[int]],
    ) -> None:
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in idxs:
            groups[batch[i].cached_len].append(i)
        for _cached_len, group in groups.items():
            self._batch_decode([batch[i] for i in group], group, radix, results)

    def _batch_prefill(
        self,
        seqs: List[Sequence],
        idxs: List[int],
        radix: RadixCache,
        results: List[Optional[int]],
    ) -> None:
        if not seqs:
            return
        news = [s.input_ids[s.cached_len :] for s in seqs]
        past_lens = [s.cached_len for s in seqs]
        new_lens = [len(n) for n in news]
        # Identical past + new lengths required for a correct single forward (no pad KV pollution)
        if len(set(past_lens)) > 1 or len(set(new_lens)) > 1:
            for s, i in zip(seqs, idxs):
                results[i] = self._prefill_one(s, radix)
            return

        B = len(seqs)
        max_new = new_lens[0]
        if max_new == 0:
            for b, (s, i) in enumerate(zip(seqs, idxs)):
                if s.last_logits is None:
                    raise RuntimeError("exact hit without last_logits")
                if self._v4_hybrid:
                    self._v4_ensure_restored(s, batch_slot=b)
                results[i] = self._sample(s.last_logits.to(self.device), s)
                s.cached_len = len(s.input_ids)
            return

        input_ids = torch.tensor(news, dtype=torch.long, device=self.device)
        max_past = past_lens[0]

        if self._v4_hybrid:
            # Official decode path assumes seqlen==1 when start_pos>0.
            if max_past > 0 and max_new > 1:
                for s, i in zip(seqs, idxs):
                    results[i] = self._prefill_one(s, radix)
                return
            for b, seq in enumerate(seqs):
                self._v4_ensure_restored(seq, batch_slot=b)
            logits = self._model_forward_v4(input_ids, start_pos=max_past)
            for b, (seq, i) in enumerate(zip(seqs, idxs)):
                nlen = new_lens[b]
                row = logits[b]
                seq.last_logits = row.detach().float().cpu().clone()
                seq.kv_state = None
                seq.cached_len = seq.cached_len + nlen
                seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + nlen
                results[i] = self._sample(row, seq)
                self._v4_maybe_save_prefix(seq, batch_slot=b)
            return

        for s in seqs:
            self._ensure_blocks(s, radix, s.cached_len + max_new)

        if self._use_paged_attention():
            outputs = self._model_forward_paged(input_ids, seqs, radix, is_decode=False)
            for b, (seq, i) in enumerate(zip(seqs, idxs)):
                nlen = new_lens[b]
                logits = outputs.logits[b, nlen - 1, :]
                start = seq.cached_len
                seq.last_logits = logits.detach().float().cpu().clone()
                seq.kv_state = None
                seq.cached_len = start + nlen
                seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + nlen
                results[i] = self._sample(logits, seq)
            return

        pasts = [self._past_for_seq(s, radix) for s in seqs]
        batched_past = self._batch_caches(pasts) if past_lens[0] > 0 else None
        attn = torch.ones((B, max_past + max_new), dtype=torch.long, device=self.device)
        outputs = self._model_forward(input_ids, batched_past, attn)
        for b, (seq, i) in enumerate(zip(seqs, idxs)):
            nlen = new_lens[b]
            logits = outputs.logits[b, nlen - 1, :]
            full_kv = self._split_batch_cache(outputs.past_key_values, b, B)
            start = seq.cached_len
            write_kv = self._slice_kv_tail(self._to_legacy_kv(full_kv), nlen)
            self._commit_pages(seq, radix, start, write_kv)
            seq.last_logits = logits.detach().float().cpu().clone()
            seq.kv_state = None if self.use_paged_as_source else full_kv
            seq.cached_len = start + nlen
            seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + nlen
            results[i] = self._sample(logits, seq)

    def _batch_decode(
        self,
        seqs: List[Sequence],
        idxs: List[int],
        radix: RadixCache,
        results: List[Optional[int]],
    ) -> None:
        if not seqs:
            return
        # All seqs in this group share cached_len
        pending_seqs: List[Sequence] = []
        pending_idxs: List[int] = []
        for b, (s, i) in enumerate(zip(seqs, idxs)):
            if not s.output_ids and getattr(s, "last_logits", None) is not None:
                if self._v4_hybrid:
                    self._v4_ensure_restored(s, batch_slot=b)
                results[i] = self._sample(s.last_logits.to(self.device), s)
                s.cached_len = len(s.input_ids)
                continue
            pending_seqs.append(s)
            pending_idxs.append(i)
        if not pending_seqs:
            return
        seqs, idxs = pending_seqs, pending_idxs
        B = len(seqs)
        lasts = [s.output_ids[-1] if s.output_ids else s.input_ids[-1] for s in seqs]
        input_ids = torch.tensor([[t] for t in lasts], dtype=torch.long, device=self.device)

        if self._v4_hybrid:
            for b, seq in enumerate(seqs):
                self._v4_ensure_restored(seq, batch_slot=b)
            start_pos = seqs[0].cached_len
            logits = self._model_forward_v4(input_ids, start_pos=start_pos)
            for b, (seq, i) in enumerate(zip(seqs, idxs)):
                row = logits[b]
                seq.kv_state = None
                seq.cached_len = seq.cached_len + 1
                results[i] = self._sample(row, seq)
            return

        # COW last page before append
        for s in seqs:
            pos = s.cached_len
            self._ensure_blocks(s, radix, pos + 1)
            page_i = pos // radix.block_size
            if page_i < len(s.block_table):
                s.block_table[page_i] = radix.cow_block_if_shared(s.block_table[page_i])

        if self._use_paged_attention():
            outputs = self._model_forward_paged(input_ids, seqs, radix, is_decode=True)
            for b, (seq, i) in enumerate(zip(seqs, idxs)):
                logits = outputs.logits[b, -1, :]
                seq.kv_state = None
                seq.cached_len = seq.cached_len + 1
                results[i] = self._sample(logits, seq)
            return

        pasts = [self._past_for_seq(s, radix) for s in seqs]
        batched_past = self._batch_caches(pasts)
        past_len = seqs[0].cached_len
        attn = torch.ones((B, past_len + 1), dtype=torch.long, device=self.device)
        outputs = self._model_forward(input_ids, batched_past, attn)

        for b, (seq, i) in enumerate(zip(seqs, idxs)):
            logits = outputs.logits[b, -1, :]
            full_kv = self._split_batch_cache(outputs.past_key_values, b, B)
            pos = seq.cached_len
            write_kv = self._slice_kv_tail(self._to_legacy_kv(full_kv), 1)
            self._commit_pages(seq, radix, pos, write_kv)
            seq.kv_state = None if self.use_paged_as_source else full_kv
            seq.cached_len = pos + 1
            results[i] = self._sample(logits, seq)

    def _prefill_one(self, seq: Sequence, radix: RadixCache) -> int:
        """Serial prefill fallback (different past lengths in a group)."""
        prompt = seq.input_ids
        start = seq.cached_len
        new_tokens = prompt[start:]
        if not new_tokens:
            if seq.last_logits is None:
                raise RuntimeError("exact hit without last_logits")
            if self._v4_hybrid:
                self._v4_ensure_restored(seq, batch_slot=0)
            return self._sample(seq.last_logits.to(self.device), seq)
        input_ids = torch.tensor([new_tokens], dtype=torch.long, device=self.device)
        if self._v4_hybrid:
            self._v4_ensure_restored(seq, batch_slot=0)
            if start > 0 and len(new_tokens) > 1:
                row = None
                for t_i, tok in enumerate(new_tokens):
                    step_ids = torch.tensor(
                        [[tok]], dtype=torch.long, device=self.device
                    )
                    logits = self._model_forward_v4(step_ids, start_pos=start + t_i)
                    row = logits[0]
                assert row is not None
            else:
                logits = self._model_forward_v4(input_ids, start_pos=start)
                row = logits[0]
            seq.last_logits = row.detach().float().cpu().clone()
            seq.kv_state = None
            seq.cached_len = len(prompt)
            seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + len(new_tokens)
            tok = self._sample(row, seq)
            self._v4_maybe_save_prefix(seq, batch_slot=0)
            return tok
        self._ensure_blocks(seq, radix, len(prompt))
        if self._use_paged_attention():
            outputs = self._model_forward_paged(input_ids, [seq], radix, is_decode=False)
            logits = outputs.logits[0, -1, :]
            seq.last_logits = logits.detach().float().cpu().clone()
            seq.kv_state = None
            seq.cached_len = len(prompt)
            seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + len(new_tokens)
            return self._sample(logits, seq)
        past = self._past_for_seq(seq, radix)
        attn = self._attention_mask(input_ids, past, past_len_hint=start)
        outputs = self._model_forward(input_ids, past, attn)
        logits = outputs.logits[0, -1, :]
        new_kv = outputs.past_key_values
        write_kv = self._slice_kv_tail(self._to_legacy_kv(new_kv), len(new_tokens))
        self._commit_pages(seq, radix, start, write_kv)
        seq.last_logits = logits.detach().float().cpu().clone()
        seq.kv_state = None if self.use_paged_as_source else new_kv
        seq.cached_len = len(prompt)
        seq.prefill_tokens = getattr(seq, "prefill_tokens", 0) + len(new_tokens)
        return self._sample(logits, seq)

    def _decode_one(self, seq: Sequence, radix: RadixCache) -> int:
        if not seq.output_ids and getattr(seq, "last_logits", None) is not None:
            if self._v4_hybrid:
                self._v4_ensure_restored(seq, batch_slot=0)
            seq.cached_len = len(seq.input_ids)
            return self._sample(seq.last_logits.to(device=self.device), seq)
        last_token = seq.output_ids[-1] if seq.output_ids else seq.input_ids[-1]
        input_ids = torch.tensor([[last_token]], dtype=torch.long, device=self.device)
        pos = seq.cached_len
        if self._v4_hybrid:
            self._v4_ensure_restored(seq, batch_slot=0)
            logits = self._model_forward_v4(input_ids, start_pos=pos)
            row = logits[0]
            seq.kv_state = None
            seq.cached_len = pos + 1
            return self._sample(row, seq)
        self._ensure_blocks(seq, radix, pos + 1)
        if seq.block_table:
            page_i = pos // radix.block_size
            if page_i < len(seq.block_table):
                seq.block_table[page_i] = radix.cow_block_if_shared(seq.block_table[page_i])
        if self._use_paged_attention():
            outputs = self._model_forward_paged(input_ids, [seq], radix, is_decode=True)
            logits = outputs.logits[0, -1, :]
            seq.kv_state = None
            seq.cached_len = pos + 1
            return self._sample(logits, seq)
        past = self._past_for_seq(seq, radix)
        attn = self._attention_mask(input_ids, past, past_len_hint=pos)
        outputs = self._model_forward(input_ids, past, attn)
        logits = outputs.logits[0, -1, :]
        new_kv = outputs.past_key_values
        write_kv = self._slice_kv_tail(self._to_legacy_kv(new_kv), 1)
        self._commit_pages(seq, radix, pos, write_kv)
        seq.kv_state = None if self.use_paged_as_source else new_kv
        seq.cached_len = pos + 1
        return self._sample(logits, seq)

    def _use_paged_attention(self) -> bool:
        if self._v4_hybrid:
            return False
        return bool(
            self._is_real
            and getattr(self.kernel_backend, "supports_paged_attention", False)
        )

    def _model_forward_v4(self, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Official Transformer.forward(input_ids, start_pos) → logits [B, vocab]."""
        self.model_forward_count += 1
        self.last_model_forward_size = int(input_ids.shape[0])
        out = self.model(input_ids, start_pos=int(start_pos))
        if isinstance(out, torch.Tensor):
            logits = out
        else:
            logits = out.logits
        # Official returns last-position logits [B, vocab]; accept [B, T, vocab] too.
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        return logits

    def v4_match_prefix(self, token_ids: List[int]):
        """Match ``token_ids`` against the Hybrid V4 prefix snapshot store."""
        cache = self._v4_prefix_cache
        if cache is None:
            return 0, None
        return cache.match(token_ids)

    def _v4_ensure_restored(self, seq: Sequence, *, batch_slot: int = 0) -> None:
        """Restore prefix snapshot, or clear the slot on a cold prefill."""
        if getattr(seq, "_v4_kv_pending_restore", False):
            entry = getattr(seq, "_v4_prefix_entry", None)
            seq._v4_kv_pending_restore = False
            if entry is None or self.model is None:
                return
            from .v4_prefix_cache import restore_v4_kv

            restore_v4_kv(self.model, entry.buffers, batch_slot=batch_slot)
            seq._v4_slot_prepared = True
            return
        # Cold path: wipe stale compressor / indexer state before start_pos=0.
        if (
            seq.cached_len == 0
            and not getattr(seq, "_v4_slot_prepared", False)
            and self.model is not None
        ):
            from .v4_prefix_cache import clear_v4_kv_slot

            clear_v4_kv_slot(self.model, batch_slot=batch_slot)
            seq._v4_slot_prepared = True

    def v4_release_seq(self, seq: Sequence, *, batch_slot: int = 0) -> None:
        """Drop in-GPU slot ownership after a sequence finishes or cancels."""
        if not self._v4_hybrid or self.model is None:
            return
        from .v4_prefix_cache import clear_v4_kv_slot

        clear_v4_kv_slot(self.model, batch_slot=batch_slot)
        seq._v4_prefix_entry = None
        seq._v4_kv_pending_restore = False
        seq._v4_slot_prepared = False

    def _v4_maybe_save_prefix(self, seq: Sequence, *, batch_slot: int = 0) -> None:
        """Snapshot official KV after a completed prompt prefill."""
        cache = self._v4_prefix_cache
        if cache is None or self.model is None:
            return
        if seq.cached_len != len(seq.input_ids):
            return
        from .v4_prefix_cache import snapshot_v4_kv

        buffers = snapshot_v4_kv(self.model, batch_slot=batch_slot)
        if not buffers:
            return
        cache.insert(
            seq.input_ids[: seq.cached_len],
            last_logits=seq.last_logits,
            buffers=buffers,
        )

    def _model_forward_paged(
        self,
        input_ids: torch.Tensor,
        seqs: List[Sequence],
        radix: RadixCache,
        is_decode: bool,
    ):
        """Forward with FlashInfer paged attention; no DynamicCache rebuild."""
        self.model_forward_count += 1
        self.last_model_forward_size = int(input_ids.shape[0])
        B, q_len = input_ids.shape
        cached_lens = [s.cached_len for s in seqs]
        q_lens = [q_len] * B
        past_len = cached_lens[0]
        position_ids = (
            torch.arange(past_len, past_len + q_len, device=self.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(B, -1)
        )
        attn = torch.ones((B, past_len + q_len), dtype=torch.long, device=self.device)
        n_heads = int(self.model.config.num_attention_heads)
        sm_scale = float(self.head_dim**-0.5)
        ctx = PagedAttnContext(
            radix=radix,
            block_tables=[list(s.block_table) for s in seqs],
            cached_lens=cached_lens,
            q_lens=q_lens,
            is_decode=is_decode,
            num_qo_heads=n_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            sm_scale=sm_scale,
        )
        self.kernel_backend.begin_forward(ctx)
        try:
            return self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                attention_mask=attn,
            )
        finally:
            self.kernel_backend.end_forward()

    def _model_forward(self, input_ids, past, attention_mask):
        self.model_forward_count += 1
        self.last_model_forward_size = int(input_ids.shape[0])
        return self.model(
            input_ids=input_ids,
            past_key_values=self._as_model_cache(past),
            use_cache=True,
            attention_mask=attention_mask,
        )

    def _past_for_seq(self, seq: Sequence, radix: RadixCache):
        """CPU/stub path: rebuild HF cache from paged KV when enabled."""
        if self.use_paged_as_source and seq.block_table and seq.cached_len > 0:
            self.paged_rebuild_count += 1
            return radix.build_cache(seq.block_table, seq.cached_len)
        return self._as_model_cache(seq.kv_state)

    def _commit_pages(
        self, seq: Sequence, radix: RadixCache, start: int, write_kv: PastKV
    ) -> None:
        if not write_kv:
            raise RuntimeError(
                "empty KV write: past_key_values could not be converted to legacy "
                "(check Transformers DynamicCache / Cache API compatibility)"
            )
        append_len = write_kv[0][0].shape[-2] if write_kv[0][0].dim() == 4 else write_kv[0][0].shape[0]
        self._ensure_blocks(seq, radix, start + append_len)
        self.kernel_backend.append_paged_kv(radix, seq.block_table, start, write_kv)

    def _batch_caches(self, pasts: List):
        """Stack per-seq caches into one batched DynamicCache/legacy list."""
        if not pasts or all(p is None for p in pasts):
            return None
        legacies = []
        for p in pasts:
            if p is None:
                raise RuntimeError("cannot batch None past with non-None peers")
            leg = self._to_legacy_kv(p)
            if not leg:
                raise RuntimeError("empty KV in batch_caches")
            legacies.append(leg)
        # All same seq length (caller guarantees)
        n_layers = len(legacies[0])
        batched = []
        for layer in range(n_layers):
            ks = torch.cat([leg[layer][0] for leg in legacies], dim=0)
            vs = torch.cat([leg[layer][1] for leg in legacies], dim=0)
            batched.append((ks, vs))
        return self._as_model_cache(batched)

    def _split_batch_cache(self, past, index: int, batch_size: int):
        legacy = self._to_legacy_kv(past)
        if legacy is None:
            return None
        sliced = []
        for k, v in legacy:
            sliced.append((k[index : index + 1].contiguous(), v[index : index + 1].contiguous()))
        return self._as_model_cache(sliced)

    def _sample(self, logits: torch.Tensor, seq: Sequence) -> int:
        temperature = float(getattr(seq, "temperature", 0.0) or 0.0)
        top_p = float(getattr(seq, "top_p", 1.0) or 1.0)
        top_k = getattr(seq, "top_k", None)
        seed = getattr(seq, "seed", None)
        step = len(seq.output_ids)
        gen = make_generator(self.device, int(seed) + step) if seed is not None else None
        return sample_logits(
            logits, temperature=temperature, top_p=top_p, top_k=top_k, generator=gen
        )

    def _ensure_blocks(self, seq: Sequence, radix: RadixCache, total_len: int) -> None:
        needed = (total_len + radix.block_size - 1) // radix.block_size
        while len(seq.block_table) < needed:
            seq.block_table.extend(radix.allocate_blocks(1))

    def _slice_kv_tail(self, kv: Optional[PastKV], n: int) -> PastKV:
        if not kv:
            return []
        out = []
        for k, v in kv:
            if k.dim() == 4:
                out.append((k[:, :, -n:, :].contiguous(), v[:, :, -n:, :].contiguous()))
            elif k.dim() == 3:
                out.append((k[-n:, :, :].contiguous(), v[-n:, :, :].contiguous()))
            else:
                out.append((k, v))
        return out

    def _attention_mask(
        self, input_ids: torch.Tensor, past, past_len_hint: Optional[int] = None
    ) -> torch.Tensor:
        past_len = 0
        if past_len_hint is not None:
            past_len = past_len_hint
        elif past is not None and hasattr(past, "get_seq_length"):
            past_len = int(past.get_seq_length())
        elif isinstance(past, (list, tuple)) and past:
            k0 = past[0][0]
            past_len = int(k0.shape[-2]) if k0.dim() >= 3 else 0
        total = past_len + input_ids.shape[-1]
        return torch.ones((input_ids.shape[0], total), dtype=torch.long, device=input_ids.device)

    def _cast_legacy_kv(self, past: PastKV) -> PastKV:
        """Ensure K/V match model dtype (avoids bf16×fp16 → fp32 SDPA promotion)."""
        dt = self.torch_dtype
        out: PastKV = []
        for k, v in past:
            if k.dtype != dt:
                k = k.to(dtype=dt)
            if v.dtype != dt:
                v = v.to(dtype=dt)
            out.append((k, v))
        return out

    def _as_model_cache(self, past):
        if past is None:
            return None
        if isinstance(past, (list, tuple)):
            past = self._cast_legacy_kv(list(past))
            try:
                from transformers import DynamicCache

                if hasattr(DynamicCache, "from_legacy_cache"):
                    return DynamicCache.from_legacy_cache(past)
            except Exception:
                pass
            try:
                from transformers import DynamicCache

                cache = DynamicCache()
                for layer_idx, (k, v) in enumerate(past):
                    cache.update(k, v, layer_idx)
                return cache
            except Exception:
                return past
        # Existing Cache object: rebuild if dtype mismatches model
        if hasattr(past, "get_seq_length") or hasattr(past, "layers"):
            legacy = self._to_legacy_kv(past)
            if legacy and any(k.dtype != self.torch_dtype for k, _ in legacy):
                return self._as_model_cache(legacy)
            return past
        return past

    def _to_legacy_kv(self, past) -> Optional[PastKV]:
        """Normalize HF cache → list[(k,v)] for page commit.

        Transformers 4.x: DynamicCache.to_legacy_cache().
        Transformers 5.x: to_legacy_cache may be absent; use layers[i].keys/values
        or Cache.__getitem__ instead.
        """
        if past is None:
            return None
        if isinstance(past, (list, tuple)):
            if not past:
                return None
            if isinstance(past[0], (list, tuple)) and len(past[0]) >= 2:
                return list(past)
            return None
        # Prefer official conversion when present and non-empty.
        if hasattr(past, "to_legacy_cache"):
            try:
                legacy = past.to_legacy_cache()
                if legacy:
                    return list(legacy)
            except Exception:
                pass
        # Transformers 5+ / Cache API: per-layer .keys / .values
        layers = getattr(past, "layers", None)
        if layers is not None:
            out: PastKV = []
            for layer in layers:
                k = getattr(layer, "keys", None)
                v = getattr(layer, "values", None)
                if k is None or v is None:
                    continue
                out.append((k, v))
            if out:
                return out
        # Cache protocol: past[i] -> (k, v)
        try:
            n = len(past)
        except Exception:
            n = 0
        if n > 0:
            out = []
            for i in range(n):
                try:
                    pair = past[i]
                except Exception:
                    break
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    out.append((pair[0], pair[1]))
            if out:
                return out
        return None

    def sample_next(
        self,
        logits: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> int:
        gen = make_generator(self.device, seed)
        return sample_logits(
            logits, temperature=temperature, top_p=top_p, top_k=top_k, generator=gen
        )


MoEModelRunner = ModelRunner
