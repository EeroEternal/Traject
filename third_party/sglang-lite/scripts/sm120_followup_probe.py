#!/usr/bin/env python3
"""SM120 follow-up probe for docs/deepseek-v4-flash-plan.md §3.0.3 checklist.

Run on RTX 5090:
  CUDA_VISIBLE_DEVICES=0 python scripts/sm120_followup_probe.py
"""

from __future__ import annotations

import importlib
import inspect
import json
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class Item:
    name: str
    ok: bool = False
    result: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


def env() -> Dict[str, str]:
    import torch

    maj, min_ = torch.cuda.get_device_capability(0)
    import flashinfer

    return {
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
        "capability": f"({maj}, {min_})",
        "flashinfer": getattr(flashinfer, "__version__", "?"),
    }


def check_sparse_mla_sm120() -> Item:
    item = Item(name="SM120 sparse MLA entry")
    try:
        import flashinfer
        import flashinfer.mla as mla
        import torch

        detail: Dict[str, Any] = {}
        # Presence of internal / public SM120 sparse symbols
        for name in (
            "flashinfer.mla._sparse_mla_sm120",
            "flashinfer.mla.sparse_mla_sm120",
        ):
            try:
                mod = importlib.import_module(name)
                detail[name] = [a for a in dir(mod) if "sm120" in a.lower() or "dsv4" in a.lower()][
                    :20
                ]
            except Exception as e:
                detail[name] = f"{type(e).__name__}: {e}"

        fn = getattr(mla, "trtllm_batch_decode_sparse_mla_dsv4", None)
        detail["has_dsv4_api"] = fn is not None
        if fn is not None:
            src = inspect.getsource(fn)
            detail["dsv4_mentions_sm120"] = any(
                k in src.lower() for k in ("sm120", "sm12", "sparse backend", "packed sparse")
            )
            # Also check signature for backend kw
            try:
                detail["dsv4_sig"] = str(inspect.signature(fn))
            except Exception as e:
                detail["dsv4_sig_err"] = str(e)

        # Run DSv4 sparse — FI≥0.6.16 routes to SM120 with packed uint8 584 layout
        B, qlen, H = 1, 1, 64
        page_size, swa_pages, comp_pages = 64, 4, 16
        swa_topk, sparse_topk = 128, 128
        packed_dim = 584
        q = torch.randn(B, qlen, H, 512, device="cuda", dtype=torch.bfloat16)
        swa_kv = torch.zeros(swa_pages, 1, page_size, packed_dim, device="cuda", dtype=torch.uint8)
        compressed = torch.zeros(comp_pages, 1, page_size, packed_dim, device="cuda", dtype=torch.uint8)
        swa_idx = torch.randint(
            0, swa_pages * page_size, (B * qlen, swa_topk), device="cuda", dtype=torch.int32
        )
        comp_idx = torch.randint(
            0, comp_pages * page_size, (B * qlen, sparse_topk), device="cuda", dtype=torch.int32
        )
        swa_topk_lens = torch.full((B * qlen,), swa_topk, device="cuda", dtype=torch.int32)
        comp_topk_lens = torch.full((B * qlen,), sparse_topk, device="cuda", dtype=torch.int32)
        seq_lens = torch.full((B,), swa_pages * page_size, device="cuda", dtype=torch.int32)
        workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        try:
            kwargs = dict(
                query=q,
                kv_cache=swa_kv,
                workspace_buffer=workspace,
                sparse_indices=swa_idx,
                compressed_kv_cache=compressed,
                sparse_topk_lens=None,
                seq_lens=seq_lens,
                kv_layout="HND",
            )
            # Newer FI accepts swa / extra sparse kwargs
            try:
                out = mla.trtllm_batch_decode_sparse_mla_dsv4(
                    **kwargs,
                    swa_topk_lens=swa_topk_lens,
                    extra_sparse_indices=comp_idx,
                    extra_sparse_topk_lens=comp_topk_lens,
                )
            except TypeError:
                out = mla.trtllm_batch_decode_sparse_mla_dsv4(
                    q,
                    swa_kv,
                    workspace,
                    swa_idx,
                    compressed,
                    comp_topk_lens,
                    seq_lens,
                    kv_layout="HND",
                )
            detail["run"] = {
                "shape": list(out.shape),
                "finite": bool(torch.isfinite(out).all().item()),
            }
            item.ok = bool(detail["run"]["finite"])
            item.result = f"dsv4 sparse MLA ran on sm_120; out={detail['run']['shape']}"
        except Exception as e:
            detail["run_error"] = f"{type(e).__name__}: {e}"
            detail["run_tb"] = traceback.format_exc()[-1200:]
            item.ok = False
            item.result = f"dsv4 sparse MLA failed: {type(e).__name__}: {e}"

        # XQA MLA is SM120 dense path — record availability
        xqa = getattr(mla, "trtllm_batch_decode_with_kv_cache_mla", None) or getattr(
            flashinfer, "xqa_batch_decode_with_kv_cache_mla", None
        )
        detail["xqa_or_mla_decode_api"] = bool(xqa is not None)
        item.detail = detail
        if not item.ok and not detail.get("dsv4_mentions_sm120"):
            item.result += " | wheel likely lacks SM120 sparse dispatch (need FlashInfer≥PR#3395 / 0.6.16+)"
        return item
    except Exception as e:
        item.ok = False
        item.result = f"{type(e).__name__}: {e}"
        item.detail["tb"] = traceback.format_exc()[-1500:]
        return item


def check_b12x_moe() -> Item:
    item = Item(name="B12xMoEWrapper smoke")
    try:
        import flashinfer
        import torch

        Wrapper = flashinfer.B12xMoEWrapper
        item.detail["init_sig"] = str(inspect.signature(Wrapper.__init__))
        # Tiny MoE: construct + run with packed FP4 + 6D MMA scales
        import math

        from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

        num_experts, top_k, hidden, inter = 8, 2, 256, 512
        try:
            w = Wrapper(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden,
                intermediate_size=inter,
                max_num_tokens=64,
                device="cuda",
                output_dtype=torch.bfloat16,
                quant_mode="nvfp4",
            )
            item.detail["constructed"] = True
            item.detail["wrapper_type"] = type(w).__name__
            methods = [a for a in dir(w) if not a.startswith("_") and callable(getattr(w, a))]
            item.detail["methods"] = methods[:40]

            def _sf_mma(m: int, k: int, e: int) -> torch.Tensor:
                m_tiles = math.ceil(m / 128)
                k_tiles = math.ceil(math.ceil(k / 16) / 4)
                flat = torch.randint(
                    1, 120, (e * m_tiles * k_tiles * 32 * 4 * 4,), device="cuda", dtype=torch.uint8
                )
                return convert_sf_to_mma_layout(flat, m=m, k=k, num_groups=e)

            n_tok = 4
            w1q = torch.randint(0, 255, (num_experts, 2 * inter, hidden // 2), device="cuda", dtype=torch.uint8)
            w2q = torch.randint(0, 255, (num_experts, hidden, inter // 2), device="cuda", dtype=torch.uint8)
            w1sf = _sf_mma(2 * inter, hidden, num_experts)
            w2sf = _sf_mma(hidden, inter, num_experts)
            x = torch.randn(n_tok, hidden, device="cuda", dtype=torch.bfloat16)
            experts = torch.randint(0, num_experts, (n_tok, top_k), device="cuda", dtype=torch.int32)
            scales = torch.softmax(torch.randn(n_tok, top_k, device="cuda"), dim=-1).float()
            ones_e = torch.ones(num_experts, device="cuda", dtype=torch.float32)
            out = w.run(
                x,
                w1q,
                w1sf,
                w2q,
                w2sf,
                experts,
                scales,
                w1_alpha=ones_e,
                w2_alpha=ones_e,
                fc2_input_scale=torch.ones(1, device="cuda", dtype=torch.float32),
            )
            finite = bool(torch.isfinite(out).all().item())
            item.detail["run"] = {"shape": list(out.shape), "finite": finite}
            item.ok = finite
            item.result = f"B12xMoEWrapper run ok shape={list(out.shape)}; HF rel-err not closed"
        except Exception as e:
            item.detail["construct_error"] = f"{type(e).__name__}: {e}"
            item.detail["construct_tb"] = traceback.format_exc()[-1200:]
            item.ok = False
            item.result = f"B12xMoEWrapper importable but construct/run failed: {e}"
        # Also try functional b12x_fused_moe signature
        fn = getattr(flashinfer, "b12x_fused_moe", None)
        if fn is not None:
            try:
                item.detail["b12x_fused_moe_sig"] = str(inspect.signature(fn))
            except Exception as e:
                item.detail["b12x_fused_moe_sig_err"] = str(e)
        return item
    except Exception as e:
        item.ok = False
        item.result = f"{type(e).__name__}: {e}"
        item.detail["tb"] = traceback.format_exc()[-1500:]
        return item


def check_sm120_gemm_and_deepgemm() -> Item:
    item = Item(name="SM120 GEMM / DeepGEMM")
    try:
        import flashinfer
        import torch

        detail: Dict[str, Any] = {}
        # FlashInfer Sm120 B12x dense gemm class
        cls = getattr(flashinfer.gemm, "Sm120B12xBlockScaledDenseGemmKernel", None)
        detail["has_Sm120B12xBlockScaledDenseGemmKernel"] = cls is not None
        if cls is not None:
            detail["sm120_gemm_attrs"] = [a for a in dir(cls) if not a.startswith("_")][:30]

        # standalone deep_gemm
        try:
            import deep_gemm as dg

            detail["deep_gemm_version"] = getattr(dg, "__version__", "?")
            detail["deep_gemm_arch"] = (
                dg.get_cuda_arch() if hasattr(dg, "get_cuda_arch") else None
            )
            a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            d = torch.zeros(256, 256, device="cuda", dtype=torch.bfloat16)
            try:
                dg.bf16_gemm_nt(a, b, d)
                ref = a @ b.t()
                err = (d.float() - ref.float()).abs().max().item()
                detail["deep_gemm_bf16"] = {"ok": True, "max_abs": err}
            except Exception as e:
                detail["deep_gemm_bf16"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        except ImportError as e:
            detail["deep_gemm_import"] = str(e)

        # flashinfer.deep_gemm — check for SM120 runtimes
        try:
            import flashinfer.deep_gemm as fidg

            attrs = [a for a in dir(fidg) if "SM120" in a or "sm120" in a or "SM100" in a]
            detail["flashinfer_deep_gemm_arch_attrs"] = attrs
        except Exception as e:
            detail["flashinfer_deep_gemm"] = str(e)

        dg_ok = bool(detail.get("deep_gemm_bf16", {}).get("ok"))
        has_b12x = bool(detail["has_Sm120B12xBlockScaledDenseGemmKernel"])
        item.ok = has_b12x  # fallback path available even if deep_gemm fails
        item.result = (
            f"Sm120B12xGemm={'yes' if has_b12x else 'no'}; "
            f"deep_gemm_bf16={'ok' if dg_ok else 'FAIL'}; "
            f"fallback={'B12x/sgl-kernel' if has_b12x and not dg_ok else 'n/a'}"
        )
        item.detail = detail
        return item
    except Exception as e:
        item.ok = False
        item.result = f"{type(e).__name__}: {e}"
        item.detail["tb"] = traceback.format_exc()[-1500:]
        return item


def check_standard_mla_still_ok() -> Item:
    """Regression: BatchMLAPagedAttentionWrapper still works after upgrades."""
    item = Item(name="standard BatchMLAPagedAttentionWrapper")
    try:
        import flashinfer
        import torch

        num_local_heads = 8
        batch_size = 2
        head_dim_ckv = 512
        head_dim_kpe = 64
        page_size = 1
        kv_len = 32
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
        q_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda")
        kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device="cuda")
        kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device="cuda") * kv_len
        kv_indices = torch.arange(0, batch_size * kv_len, dtype=torch.int32, device="cuda")
        q_nope = torch.randn(batch_size, num_local_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda")
        q_pe = torch.randn(batch_size, num_local_heads, head_dim_kpe, dtype=torch.bfloat16, device="cuda")
        ckv = torch.randn(batch_size * kv_len, page_size, head_dim_ckv, dtype=torch.bfloat16, device="cuda")
        kpe = torch.randn(batch_size * kv_len, page_size, head_dim_kpe, dtype=torch.bfloat16, device="cuda")
        sm_scale = 1.0 / ((128 + 64) ** 0.5)
        wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_lens,
            num_local_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            True,
            sm_scale,
            q_nope.dtype,
            ckv.dtype,
        )
        out = wrapper.run(q_nope, q_pe, ckv, kpe)
        item.ok = bool(torch.isfinite(out).all().item())
        item.result = f"ok shape={list(out.shape)}"
        item.detail = {"shape": list(out.shape)}
        return item
    except Exception as e:
        item.ok = False
        item.result = f"{type(e).__name__}: {e}"
        item.detail["tb"] = traceback.format_exc()[-1200:]
        return item


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("CUDA required")
        return 2
    items: List[Item] = [
        check_standard_mla_still_ok(),
        check_sparse_mla_sm120(),
        check_b12x_moe(),
        check_sm120_gemm_and_deepgemm(),
    ]
    out = {"env": env(), "items": [asdict(i) for i in items]}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n## SUMMARY")
    for i in items:
        print(f"- [{'OK' if i.ok else 'NO'}] {i.name}: {i.result}")
    return 0 if all(i.ok for i in items) else 1


if __name__ == "__main__":
    raise SystemExit(main())
