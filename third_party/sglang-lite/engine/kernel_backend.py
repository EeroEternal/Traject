"""KernelBackend: the single hardware abstraction point for kernel-level ops.

Everything hardware-specific (FlashInfer, CUDA-only kernels, future Ascend/CANN
backends) lives behind this interface. The rest of the engine (RadixCache,
Scheduler, ModelRunner composition) stays device-agnostic and must not import
kernel libraries or branch on device types directly.

Architecture-family routing (SM90/SM100/SM120) lives in ``capability`` and is
exposed on each backend instance — runners must not write ``if major == 10``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch

from .capability import (
    ArchFamily,
    KernelCapabilities,
    MoeGemmBackend,
    SparseMlaBackend,
    probe_kernel_capabilities,
)
from .kv_cache import PastKV, RadixCache


@dataclass
class PagedAttnContext:
    """Per-forward metadata for paged attention (set by the runner, consumed by hooks)."""

    radix: RadixCache
    block_tables: List[List[int]]
    # KV length already committed before this forward (exclusive start for new tokens).
    cached_lens: List[int]
    # Number of new query tokens per sequence in this forward.
    q_lens: List[int]
    is_decode: bool
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    sm_scale: float
    # Filled by begin_forward after planning.
    planned: bool = False


class KernelBackend:
    """Device-agnostic fallback backend (pure torch tensor copies)."""

    name = "torch"
    supports_paged_attention = False
    supports_mla = False
    supports_sparse_mla = False

    def __init__(self, capabilities: Optional[KernelCapabilities] = None):
        self.capabilities = capabilities or KernelCapabilities(
            arch_family=ArchFamily.UNKNOWN,
            cuda_capability=(0, 0),
        )

    @property
    def arch_family(self) -> ArchFamily:
        return self.capabilities.arch_family

    @property
    def sparse_mla_backend(self) -> SparseMlaBackend:
        return self.capabilities.sparse_mla_backend

    @property
    def moe_gemm_backend(self) -> MoeGemmBackend:
        return self.capabilities.moe_gemm_backend

    def append_paged_kv(
        self,
        radix: RadixCache,
        block_table: List[int],
        start: int,
        write_kv: PastKV,
    ) -> None:
        """Write per-layer K/V for tokens [start, start+n) into paged cache."""
        radix.write_kv(block_table, start, write_kv)

    def append_paged_kv_layer(
        self,
        radix: RadixCache,
        block_table: List[int],
        start: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        radix.write_kv_layer(block_table, start, layer_idx, k, v)

    def attach_to_model(self, model: Any, num_qo_heads: int, head_dim: int, sm_scale: float) -> int:
        """Optional: install attention hooks. Torch backend is a no-op."""
        return 0

    def begin_forward(self, ctx: PagedAttnContext) -> None:
        return None

    def end_forward(self) -> None:
        return None

    def paged_prefill(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        page_indices: torch.Tensor,
        page_indptr: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        sm_scale: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError("paged_prefill requires FlashInferBackend")

    def paged_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_indices: torch.Tensor,
        page_indptr: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        sm_scale: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError("paged_decode requires FlashInferBackend")

    def mla_paged_run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv: torch.Tensor,
        kpe: torch.Tensor,
        **plan_kwargs: Any,
    ) -> torch.Tensor:
        """Standard FlashInfer BatchMLA paged attention (non-sparse)."""
        raise NotImplementedError("mla_paged_run requires FlashInferBackend")

    def sparse_mla_decode_dsv4(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """DeepSeek-V4 sparse MLA decode — routed by arch_family."""
        raise NotImplementedError("sparse_mla_decode_dsv4 requires FlashInferBackend")


class FlashInferBackend(KernelBackend):
    """CUDA backend using FlashInfer paged-KV kernels."""

    name = "flashinfer"
    supports_paged_attention = True
    supports_mla = True

    def __init__(self, device: str, capabilities: Optional[KernelCapabilities] = None):
        import flashinfer

        caps = capabilities or probe_kernel_capabilities(device)
        super().__init__(caps)
        self._flashinfer = flashinfer
        self.device = device
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, kv_layout="NHD"
        )
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, kv_layout="NHD"
        )
        self._mla_wrapper = None
        try:
            self._mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                self.workspace_buffer, backend="fa2"
            )
        except Exception:
            self._mla_wrapper = None
            self.supports_mla = False
        self.supports_sparse_mla = (
            self.sparse_mla_backend == SparseMlaBackend.FLASHINFER_SPARSE_SM120
            or self.sparse_mla_backend == SparseMlaBackend.FLASHINFER_SPARSE_SM100
        )
        self._attn_ctx: Optional[PagedAttnContext] = None
        self._page_indices: Optional[torch.Tensor] = None
        self._page_indptr: Optional[torch.Tensor] = None
        self._last_page_len: Optional[torch.Tensor] = None
        self._qo_indptr: Optional[torch.Tensor] = None
        self._patched_modules: List[Any] = []

    def append_paged_kv(
        self,
        radix: RadixCache,
        block_table: List[int],
        start: int,
        write_kv: PastKV,
    ) -> None:
        first_k = write_kv[0][0]
        append_len = first_k.shape[-2] if first_k.dim() == 4 else first_k.shape[0]
        pages_after = (start + append_len + radix.block_size - 1) // radix.block_size
        if len(block_table) < pages_after:
            raise RuntimeError(
                f"append_paged_kv: need {pages_after} pages, have {len(block_table)}"
            )
        device = self.device
        for layer_idx, (k, v) in enumerate(write_kv):
            self._append_one_layer(
                radix, block_table, start, layer_idx, k, v, append_len, device
            )

    def append_paged_kv_layer(
        self,
        radix: RadixCache,
        block_table: List[int],
        start: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        k_tok, v_tok = radix.normalize_kv(k, v)
        append_len = k_tok.shape[0]
        self._append_one_layer(
            radix, block_table, start, layer_idx, k_tok, v_tok, append_len, self.device
        )

    def _append_one_layer(
        self,
        radix: RadixCache,
        block_table: List[int],
        start: int,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        append_len: int,
        device: str,
    ) -> None:
        pages_after = (start + append_len + radix.block_size - 1) // radix.block_size
        if len(block_table) < pages_after:
            raise RuntimeError(
                f"append_paged_kv_layer: need {pages_after} pages, have {len(block_table)}"
            )
        k_tok, v_tok = radix.normalize_kv(k, v)
        k_new = k_tok.to(device=device, dtype=radix.dtype)
        v_new = v_tok.to(device=device, dtype=radix.dtype)
        batch_indices = torch.zeros(append_len, dtype=torch.int32, device=device)
        positions = torch.arange(
            start, start + append_len, dtype=torch.int32, device=device
        )
        active = block_table[:pages_after]
        kv_indices = torch.tensor(active, dtype=torch.int32, device=device)
        kv_indptr = torch.tensor(
            [0, kv_indices.numel()], dtype=torch.int32, device=device
        )
        last_len = start % radix.block_size if start > 0 else 0
        kv_last = torch.tensor([last_len], dtype=torch.int32, device=device)
        self._flashinfer.append_paged_kv_cache(
            k_new,
            v_new,
            batch_indices,
            positions,
            (radix.k_cache[layer_idx], radix.v_cache[layer_idx]),
            kv_indices,
            kv_indptr,
            kv_last,
            kv_layout="NHD",
        )

    @staticmethod
    def _page_metadata(
        block_tables: Sequence[Sequence[int]],
        seq_lens: Sequence[int],
        page_size: int,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices: List[int] = []
        indptr = [0]
        last_page_lens: List[int] = []
        for bt, slen in zip(block_tables, seq_lens):
            if slen <= 0:
                raise RuntimeError("paged attention requires seq_len > 0")
            npages = (slen + page_size - 1) // page_size
            if len(bt) < npages:
                raise RuntimeError(
                    f"block_table too short: need {npages} pages for seq_len={slen}"
                )
            indices.extend(list(bt[:npages]))
            indptr.append(len(indices))
            last = slen % page_size
            last_page_lens.append(page_size if last == 0 else last)
        return (
            torch.tensor(indices, dtype=torch.int32, device=device),
            torch.tensor(indptr, dtype=torch.int32, device=device),
            torch.tensor(last_page_lens, dtype=torch.int32, device=device),
        )

    def paged_prefill(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        page_indices: torch.Tensor,
        page_indptr: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        sm_scale: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        self.prefill_wrapper.plan(
            qo_indptr,
            page_indptr,
            page_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            pos_encoding_mode="NONE",
            sm_scale=sm_scale,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        return self.prefill_wrapper.run(q, (k_cache, v_cache))

    def paged_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_indices: torch.Tensor,
        page_indptr: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        sm_scale: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        self.decode_wrapper.plan(
            page_indptr,
            page_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode="NONE",
            sm_scale=sm_scale,
            q_data_type=dtype,
            kv_data_type=dtype,
        )
        return self.decode_wrapper.run(q, (k_cache, v_cache))

    def begin_forward(self, ctx: PagedAttnContext) -> None:
        total_lens = [c + q for c, q in zip(ctx.cached_lens, ctx.q_lens)]
        page_size = ctx.radix.block_size
        self._page_indices, self._page_indptr, self._last_page_len = self._page_metadata(
            ctx.block_tables, total_lens, page_size, self.device
        )
        if ctx.is_decode:
            self._qo_indptr = None
            self.decode_wrapper.plan(
                self._page_indptr,
                self._page_indices,
                self._last_page_len,
                ctx.num_qo_heads,
                ctx.num_kv_heads,
                ctx.head_dim,
                page_size,
                pos_encoding_mode="NONE",
                sm_scale=ctx.sm_scale,
                q_data_type=ctx.radix.dtype,
                kv_data_type=ctx.radix.dtype,
            )
        else:
            qo = [0]
            for qlen in ctx.q_lens:
                qo.append(qo[-1] + qlen)
            self._qo_indptr = torch.tensor(qo, dtype=torch.int32, device=self.device)
            self.prefill_wrapper.plan(
                self._qo_indptr,
                self._page_indptr,
                self._page_indices,
                self._last_page_len,
                ctx.num_qo_heads,
                ctx.num_kv_heads,
                ctx.head_dim,
                page_size,
                causal=True,
                pos_encoding_mode="NONE",
                sm_scale=ctx.sm_scale,
                q_data_type=ctx.radix.dtype,
                kv_data_type=ctx.radix.dtype,
            )
        ctx.planned = True
        self._attn_ctx = ctx

    def end_forward(self) -> None:
        self._attn_ctx = None
        self._page_indices = None
        self._page_indptr = None
        self._last_page_len = None
        self._qo_indptr = None

    def attach_to_model(self, model: Any, num_qo_heads: int, head_dim: int, sm_scale: float) -> int:
        """Monkeypatch HF attention modules so forward uses FlashInfer paged KV."""
        patched = 0
        for module in model.modules():
            if not self._is_attention_module(module):
                continue
            if getattr(module, "_sglang_lite_paged", False):
                continue
            module._sglang_lite_orig_forward = module.forward
            module.forward = self._make_attention_forward(module)
            module._sglang_lite_paged = True
            self._patched_modules.append(module)
            patched += 1
        return patched

    @staticmethod
    def _is_attention_module(module: Any) -> bool:
        return (
            hasattr(module, "q_proj")
            and hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
            and hasattr(module, "o_proj")
            and hasattr(module, "head_dim")
            and hasattr(module, "layer_idx")
            and module.layer_idx is not None
        )

    def _make_attention_forward(self, attn_module: Any):
        backend = self
        layer_idx = int(attn_module.layer_idx)

        # Resolve apply_rotary_pos_emb from the modeling module that defined this class.
        mod = __import__(attn_module.__class__.__module__, fromlist=["apply_rotary_pos_emb"])
        apply_rotary_pos_emb = getattr(mod, "apply_rotary_pos_emb", None)

        def forward(
            hidden_states: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Any = None,
            **kwargs,
        ):
            ctx = backend._attn_ctx
            orig = attn_module._sglang_lite_orig_forward
            if ctx is None:
                return orig(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, attn_module.head_dim)

            query_states = attn_module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = attn_module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = attn_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            if position_embeddings is None:
                raise RuntimeError("paged attention requires position_embeddings")
            cos, sin = position_embeddings
            if apply_rotary_pos_emb is None:
                raise RuntimeError("apply_rotary_pos_emb not found for attention module")
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Append new K/V into paged cache, then attend against full pages.
            B, _, S, _ = key_states.shape
            for b in range(B):
                start = ctx.cached_lens[b]
                # (kv_heads, S, D) -> write_kv_layer normalizes
                backend.append_paged_kv_layer(
                    ctx.radix,
                    ctx.block_tables[b],
                    start,
                    layer_idx,
                    key_states[b : b + 1],
                    value_states[b : b + 1],
                )

            dtype = ctx.radix.dtype
            page_size = ctx.radix.block_size
            k_cache = ctx.radix.k_cache[layer_idx]
            v_cache = ctx.radix.v_cache[layer_idx]

            # FlashInfer returns head-major; HF o_proj path expects (B, S, H, D).
            if ctx.is_decode:
                # query_states: (B, H, 1, D) -> (B, H, D)
                q = query_states.squeeze(2).contiguous().to(dtype=dtype)
                out = backend.decode_wrapper.run(q, (k_cache, v_cache))
                attn_states = out.unsqueeze(1)  # (B, 1, H, D)
            else:
                # (B, H, S, D) -> (B*S, H, D) ragged-compatible equal lengths
                q = (
                    query_states.transpose(1, 2)
                    .reshape(-1, ctx.num_qo_heads, ctx.head_dim)
                    .contiguous()
                    .to(dtype=dtype)
                )
                out = backend.prefill_wrapper.run(q, (k_cache, v_cache))
                attn_states = out.view(B, S, ctx.num_qo_heads, ctx.head_dim)

            attn_output = attn_states.reshape(*input_shape, -1).contiguous()
            attn_output = attn_module.o_proj(attn_output)
            # Match HF MixtralAttention return: (attn_output, attn_weights)
            return attn_output, None

        return forward

    def mla_paged_run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv: torch.Tensor,
        kpe: torch.Tensor,
        **plan_kwargs: Any,
    ) -> torch.Tensor:
        if self._mla_wrapper is None:
            raise NotImplementedError("BatchMLAPagedAttentionWrapper unavailable")
        self._mla_wrapper.plan(**plan_kwargs)
        return self._mla_wrapper.run(q_nope, q_pe, ckv, kpe)

    def sparse_mla_decode_dsv4(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Route V4 sparse MLA by arch_family — never SM100 TRTLLM on SM120.

        FlashInfer path: keyword args match
        ``flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4``.

        Official fallback: pass ``q, kv, attn_sink, topk_idxs, softmax_scale``
        (same as TileLang ``sparse_attn``) when backend is
        ``OFFICIAL_SPARSE_ATTN`` and ``_official_sparse_attn`` was attached.
        """
        backend = self.sparse_mla_backend
        if backend == SparseMlaBackend.FLASHINFER_SPARSE_SM100 and self.arch_family == ArchFamily.SM120:
            raise RuntimeError(
                "refusing SM100 sparse MLA on SM120; upgrade FlashInfer or use "
                "official sparse_attn fallback"
            )
        if backend in (
            SparseMlaBackend.FLASHINFER_SPARSE_SM120,
            SparseMlaBackend.FLASHINFER_SPARSE_SM100,
        ):
            import flashinfer.mla as mla

            # Prefer kwargs API used by our Hybrid hook.
            if kwargs:
                return mla.trtllm_batch_decode_sparse_mla_dsv4(*args, **kwargs)
            return mla.trtllm_batch_decode_sparse_mla_dsv4(*args)
        if backend == SparseMlaBackend.OFFICIAL_SPARSE_ATTN:
            fn = getattr(self, "_official_sparse_attn", None)
            if fn is None:
                raise NotImplementedError(
                    "official sparse_attn not attached; load Hybrid model first "
                    "or set FI≥0.6.16 with SM120 sparse module"
                )
            if args and len(args) >= 5:
                return fn(*args[:5])
            return fn(
                kwargs["q"],
                kwargs["kv"],
                kwargs["attn_sink"],
                kwargs["topk_idxs"],
                kwargs["softmax_scale"],
            )
        raise NotImplementedError(f"sparse MLA backend unavailable: {backend}")


def create_kernel_backend(device: str) -> KernelBackend:
    """Pick the best backend the current hardware supports."""
    if device != "cpu":
        try:
            caps = probe_kernel_capabilities(device)
            return FlashInferBackend(device, capabilities=caps)
        except ImportError:
            pass
    return KernelBackend(capabilities=probe_kernel_capabilities("cpu"))
