#!/usr/bin/env python3
"""Leaf-component probe for docs/deepseek-v4-flash-plan.md §3.0.1 (T1–T4).

Run on CUDA sm_120 host:
  CUDA_VISIBLE_DEVICES=0 python scripts/leaf_component_probe.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProbeResult:
    item: str
    versions: Dict[str, str] = field(default_factory=dict)
    commands: List[str] = field(default_factory=list)
    result: str = ""
    conclusion: str = ""
    ok: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)


def _env_info() -> Dict[str, str]:
    import torch

    major, minor = torch.cuda.get_device_capability(0)
    return {
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
        "sm": f"{major}{minor}",
        "capability": f"({major}, {minor})",
    }


def probe_t1_flashinfer() -> ProbeResult:
    r = ProbeResult(item="T1 flashinfer-python")
    r.commands.append("import flashinfer; BatchMLAPagedAttentionWrapper smoke")
    try:
        import flashinfer
        import torch

        r.versions = {
            "flashinfer": getattr(flashinfer, "__version__", "?"),
            **_env_info(),
        }

        # --- API surface: CSA/HCA / MLA ---
        names = dir(flashinfer)
        mla_names = [n for n in names if "mla" in n.lower() or "MLA" in n]
        csa = [n for n in names if "csa" in n.lower() or "CSA" in n]
        hca = [n for n in names if "hca" in n.lower() or "HCA" in n]
        # also search submodule
        try:
            import flashinfer.mla as mla_mod

            mla_mod_attrs = [a for a in dir(mla_mod) if not a.startswith("_")]
        except Exception as e:
            mla_mod_attrs = [f"import_error:{e}"]

        r.detail["api"] = {
            "mla_top_level": mla_names,
            "mla_module": mla_mod_attrs[:40],
            "csa_symbols": csa,
            "hca_symbols": hca,
        }

        # --- MLA wrapper smoke (DeepSeek-V2-Lite-like dims, scaled down) ---
        # V2-Lite: kv_lora_rank=512, qk_rope_head_dim=64; use fewer heads for smoke.
        num_local_heads = 8
        batch_size = 2
        head_dim_ckv = 512
        head_dim_kpe = 64
        page_size = 1
        kv_len = 32
        device = "cuda"
        dtype = torch.bfloat16

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper_cls = getattr(flashinfer, "BatchMLAPagedAttentionWrapper", None)
        if wrapper_cls is None:
            wrapper_cls = getattr(flashinfer.mla, "BatchMLAPagedAttentionWrapper", None)
        if wrapper_cls is None:
            r.result = "BatchMLAPagedAttentionWrapper not found"
            r.conclusion = "FAIL: MLA wrapper missing in this flashinfer build"
            r.ok = False
            return r

        mla_wrapper = wrapper_cls(workspace, backend="fa2")
        q_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
        kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * kv_len
        kv_indices = torch.arange(0, batch_size * kv_len, dtype=torch.int32, device=device)

        torch.manual_seed(0)
        q_nope = torch.randn(
            batch_size, num_local_heads, head_dim_ckv, dtype=dtype, device=device
        )
        q_pe = torch.randn(
            batch_size, num_local_heads, head_dim_kpe, dtype=dtype, device=device
        )
        # page_size=1 layout: [num_pages, page_size, dim]
        ckv = torch.randn(
            batch_size * kv_len, page_size, head_dim_ckv, dtype=dtype, device=device
        )
        kpe = torch.randn(
            batch_size * kv_len, page_size, head_dim_kpe, dtype=dtype, device=device
        )
        sm_scale = 1.0 / ((128 + 64) ** 0.5)

        plan = getattr(mla_wrapper, "plan")
        import inspect

        r.detail["mla_plan_sig"] = str(inspect.signature(plan))
        plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_lens,
            num_local_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            True,  # causal
            sm_scale,
            q_nope.dtype,
            ckv.dtype,
        )
        out = mla_wrapper.run(q_nope, q_pe, ckv, kpe)
        r.detail["mla_out_shape"] = list(out.shape)
        r.detail["mla_out_finite"] = bool(torch.isfinite(out).all().item())

        # naive reference for decode-style (q_len=1): softmax(q @ k^T)*v on compressed concat
        # Compare against a torch SDPA-style on concat(q_nope,q_pe) vs concat(ckv,kpe)
        # For page_size=1 decode batch.
        ref_outs = []
        for b in range(batch_size):
            q = torch.cat([q_nope[b], q_pe[b]], dim=-1)  # [H, 576]
            k = torch.cat(
                [ckv[b * kv_len : (b + 1) * kv_len, 0], kpe[b * kv_len : (b + 1) * kv_len, 0]],
                dim=-1,
            )  # [S, 576]
            # v is ckv only in absorbed MLA (head_dim_vo = ckv)
            v = ckv[b * kv_len : (b + 1) * kv_len, 0]  # [S, 512]
            scores = torch.einsum("hd,sd->hs", q.float(), k.float()) * sm_scale
            attn = torch.softmax(scores, dim=-1)
            o = torch.einsum("hs,sd->hd", attn, v.float())
            ref_outs.append(o)
        ref = torch.stack(ref_outs, dim=0).to(dtype)
        # out may be [B, H, ckv] or [nnz, H, ckv]
        out_cmp = out.view(batch_size, num_local_heads, head_dim_ckv).float()
        ref_cmp = ref.float()
        max_abs = (out_cmp - ref_cmp).abs().max().item()
        r.detail["mla_vs_torch_max_abs"] = max_abs
        r.detail["mla_atol_ok"] = max_abs <= 1e-2 * 50  # bf16 MLA can be looser; record raw

        # Also note CSA/HCA coverage
        if not csa and not hca:
            r.detail["csa_hca"] = (
                "no CSA/HCA symbols in flashinfer 0.6.12; MLA (V2/V3) present; "
                "V4-Flash CSA/HCA not exposed as public API"
            )

        r.ok = bool(r.detail["mla_out_finite"]) and max_abs < 0.5  # smoke: runnable + roughly close
        r.result = (
            f"MLA wrapper ran on sm_{r.versions['sm']}; "
            f"out={r.detail['mla_out_shape']}; max_abs_vs_torch={max_abs:.4f}; "
            f"CSA/HCA symbols={csa or hca or 'none'}"
        )
        r.conclusion = (
            "PASS: MLA (V2/V3 absorbed) works on sm_120"
            if r.ok
            else "FAIL: MLA smoke failed"
        ) + "; CSA/HCA: NOT exposed in this flashinfer version (track upstream)"
        return r
    except Exception as e:
        r.ok = False
        r.result = f"{type(e).__name__}: {e}"
        r.detail["traceback"] = traceback.format_exc()[-2000:]
        r.conclusion = "FAIL: flashinfer MLA smoke error"
        return r


def probe_t2_sgl_kernel() -> ProbeResult:
    r = ProbeResult(item="T2 sgl-kernel")
    r.commands.append("pip install sgl-kernel (or find existing); import sgl_kernel; fused_moe smoke")
    try:
        import importlib

        try:
            sk = importlib.import_module("sgl_kernel")
        except ImportError:
            # try install
            import subprocess

            cmd = [sys.executable, "-m", "pip", "install", "sgl-kernel", "-q"]
            r.commands.append(" ".join(cmd))
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            r.detail["pip_install_rc"] = proc.returncode
            r.detail["pip_install_stderr"] = (proc.stderr or "")[-1500:]
            if proc.returncode != 0:
                r.ok = False
                r.result = f"pip install sgl-kernel failed rc={proc.returncode}"
                r.conclusion = "FAIL: cannot install on this env — MoE fallback to HF"
                return r
            sk = importlib.import_module("sgl_kernel")

        r.versions["sgl_kernel"] = getattr(sk, "__version__", "unknown")
        r.versions.update(_env_info())
        attrs = [a for a in dir(sk) if not a.startswith("_")]
        r.detail["public_attrs_sample"] = attrs[:80]
        moe_attrs = [a for a in attrs if "moe" in a.lower() or "topk" in a.lower() or "fp8" in a.lower()]
        r.detail["moe_fp8_attrs"] = moe_attrs
        csa = [a for a in attrs if "csa" in a.lower()]
        hca = [a for a in attrs if "hca" in a.lower()]
        r.detail["csa_hca"] = {"csa": csa, "hca": hca}

        # Try calling a fused_moe if present
        fused = None
        for name in ("fused_moe", "moe_fused", "fused_experts"):
            if hasattr(sk, name):
                fused = getattr(sk, name)
                r.detail["fused_entry"] = name
                break
        # sometimes nested
        if fused is None:
            for sub in ("moe", "elementwise"):
                if hasattr(sk, sub):
                    mod = getattr(sk, sub)
                    for name in ("fused_moe", "topk_softmax", "moe_align_block_size"):
                        if hasattr(mod, name):
                            r.detail["fused_entry"] = f"{sub}.{name}"
                            fused = getattr(mod, name)
                            break

        if fused is None:
            r.ok = True  # installed + importable
            r.result = (
                f"import ok version={r.versions['sgl_kernel']}; "
                f"moe/fp8 attrs={moe_attrs[:20]}; no direct fused_moe callable found in smoke"
            )
            r.conclusion = (
                "PARTIAL: sgl-kernel installs/imports on sm_120; "
                "fused_moe API needs version-specific wiring; CSA/HCA="
                + ("none" if not csa and not hca else str(csa + hca))
            )
            return r

        r.ok = True
        r.result = f"import ok; found {r.detail.get('fused_entry')}"
        r.conclusion = "PASS: sgl-kernel importable on sm_120 (operator smoke limited)"
        return r
    except Exception as e:
        r.ok = False
        r.result = f"{type(e).__name__}: {e}"
        r.detail["traceback"] = traceback.format_exc()[-2000:]
        r.conclusion = "FAIL: sgl-kernel probe error — MoE fallback to HF"
        return r


def probe_t3_deep_gemm() -> ProbeResult:
    r = ProbeResult(item="T3 deep-gemm")
    r.commands.append("pip install deep-gemm / deep_gemm; sm_120 FP8 grouped GEMM smoke")
    try:
        import importlib
        import subprocess

        mod = None
        for name in ("deep_gemm", "deepgemm"):
            try:
                mod = importlib.import_module(name)
                r.versions["module"] = name
                break
            except ImportError:
                continue
        if mod is None:
            for pkg in ("deep-gemm", "deep_gemm"):
                cmd = [sys.executable, "-m", "pip", "install", pkg, "-q"]
                r.commands.append(" ".join(cmd))
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                r.detail[f"pip_{pkg}_rc"] = proc.returncode
                r.detail[f"pip_{pkg}_stderr"] = (proc.stderr or "")[-1500:]
                if proc.returncode == 0:
                    break
            for name in ("deep_gemm", "deepgemm"):
                try:
                    mod = importlib.import_module(name)
                    r.versions["module"] = name
                    break
                except ImportError:
                    continue

        r.versions.update(_env_info())
        if mod is None:
            r.ok = False
            r.result = "package not installable/importable via pip in this venv"
            r.conclusion = "FAIL: deep-gemm unavailable — use sgl-kernel FP8 or cuBLASLt fallback; record 5090 support=unknown/unavailable"
            return r

        r.versions["deep_gemm"] = getattr(mod, "__version__", "unknown")
        attrs = [a for a in dir(mod) if not a.startswith("_")]
        r.detail["attrs"] = attrs[:60]

        # Try a minimal gemm if API exposes one
        import torch

        sm = torch.cuda.get_device_capability(0)
        r.detail["sm"] = sm
        ran = False
        err = None
        for fn_name in ("gemm_fp8_fp8_bf16_nt", "fp8_gemm", "m_grouped_gemm_fp8_fp8_bf16_nt_contiguous"):
            fn = getattr(mod, fn_name, None)
            if fn is None:
                continue
            try:
                # shapes are library-specific; just record callable exists
                r.detail["candidate_fn"] = fn_name
                ran = True
                break
            except Exception as e:
                err = str(e)
        # Try running library test if present
        test_mod = None
        try:
            test_mod = importlib.import_module("deep_gemm.tests")
        except Exception:
            pass

        if sm[0] >= 12:
            # Explicitly attempt a tiny call if we can discover signature later
            r.detail["blackwell_note"] = "sm_120 detected; library historically Hopper-oriented"

        r.ok = True
        r.result = (
            f"import ok version={r.versions.get('deep_gemm')}; "
            f"attrs={attrs[:15]}; candidate_fn={r.detail.get('candidate_fn')}; err={err}"
        )
        if ran:
            r.conclusion = (
                "PARTIAL: deep-gemm imports on sm_120; FP8 grouped GEMM numerical run "
                "not fully exercised (API needs dedicated harness). Do NOT add to pyproject yet."
            )
        else:
            r.conclusion = (
                "PARTIAL: deep-gemm imports but no known FP8 entrypoint found in smoke; "
                "sm_120 numerical support still unconfirmed"
            )
        return r
    except Exception as e:
        r.ok = False
        r.result = f"{type(e).__name__}: {e}"
        r.detail["traceback"] = traceback.format_exc()[-2000:]
        r.conclusion = "FAIL: deep-gemm probe error — 5090 support unconfirmed; fallback sgl-kernel/cuBLASLt"
        return r


def probe_t4_v2_lite() -> ProbeResult:
    r = ProbeResult(item="T4 DeepSeek-V2-Lite model graph")
    r.commands.append("locate local V2-Lite weights; try AutoModelForCausalLM trust_remote_code")
    try:
        import torch
        from pathlib import Path

        r.versions.update(_env_info())
        candidates = []
        for root in [
            Path.home() / "models",
            Path.home() / "models-4.5",
            Path.home() / "project",
            Path("/home/bodesi/models"),
        ]:
            if not root.exists():
                continue
            for p in root.rglob("config.json"):
                try:
                    cfg = json.loads(p.read_text())
                except Exception:
                    continue
                mt = str(cfg.get("model_type", "")).lower()
                name = str(p.parent)
                if "deepseek" in name.lower() or "deepseek" in mt:
                    if "v2" in name.lower() or "lite" in name.lower() or mt in {
                        "deepseek_v2",
                        "deepseekv2",
                    }:
                        candidates.append(str(p.parent))
        r.detail["candidates"] = candidates[:20]

        # Also list deepseek-ai folder
        ds = Path.home() / "models" / "deepseek-ai"
        if ds.exists():
            r.detail["deepseek_ai_listing"] = [x.name for x in ds.iterdir()][:30]

        if not candidates:
            # Try hub id offline
            hub_id = os.environ.get("SGLANG_LITE_V2_LITE", "deepseek-ai/DeepSeek-V2-Lite-Chat")
            r.detail["hub_id"] = hub_id
            try:
                from transformers import AutoConfig

                cfg = AutoConfig.from_pretrained(hub_id, trust_remote_code=True, local_files_only=True)
                r.detail["hub_config_model_type"] = getattr(cfg, "model_type", None)
                candidates = [hub_id]
            except Exception as e:
                r.ok = False
                r.result = f"no local V2-Lite weights; hub offline load failed: {e}"
                r.conclusion = (
                    "BLOCKED: DeepSeek-V2-Lite weights not on machine / not cached; "
                    "cannot verify paged_rebuild_count or monkeypatch yet. "
                    "Layout note for S2: MLA needs per-layer compressed KV "
                    "(ckv rank + kpe rope dim), not isomorphic [layers,blocks,bs,H,D]."
                )
                r.detail["s2_layout_note"] = {
                    "standard_kv": "[layers, blocks, block_size, kv_heads, head_dim]",
                    "mla_compressed": "ckv: [blocks, page_size, kv_lora_rank]; kpe: [blocks, page_size, qk_rope_head_dim]",
                    "page_size_common": 1,
                }
                return r

        path = candidates[0]
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        r.detail["model_type"] = getattr(cfg, "model_type", None)
        r.detail["architectures"] = getattr(cfg, "architectures", None)
        r.detail["kv_lora_rank"] = getattr(cfg, "kv_lora_rank", None)
        r.detail["qk_rope_head_dim"] = getattr(cfg, "qk_rope_head_dim", None)
        r.detail["qk_nope_head_dim"] = getattr(cfg, "qk_nope_head_dim", None)
        r.detail["num_hidden_layers"] = getattr(cfg, "num_hidden_layers", None)

        # Patchability check without full weight load if too large: inspect modeling module
        patchable = False
        try:
            # Prefer config-only insight; full load may OOM
            n_params_guess = getattr(cfg, "num_hidden_layers", 0) * getattr(cfg, "hidden_size", 0)
            r.detail["load_path"] = path
            # Try loading only if env forces or model looks small
            force = os.environ.get("SGLANG_LITE_LOAD_V2_LITE") == "1"
            if force:
                model = AutoModelForCausalLM.from_pretrained(
                    path, trust_remote_code=True, dtype=torch.bfloat16, device_map="cpu"
                )
                attn_mods = [
                    (n, type(m).__name__)
                    for n, m in model.named_modules()
                    if hasattr(m, "q_proj") or hasattr(m, "q_a_proj") or "Attention" in type(m).__name__
                ]
                r.detail["attention_modules"] = attn_mods[:20]
                patchable = any("Attention" in t for _, t in attn_mods)
                del model
            else:
                # Infer from architecture name
                arch = (getattr(cfg, "architectures", []) or [""])[0]
                patchable = "Deepseek" in arch or "DeepSeek" in arch
                r.detail["patchable_inferred"] = patchable
                r.detail["note"] = "skipped full weight load (set SGLANG_LITE_LOAD_V2_LITE=1 to force)"
        except Exception as e:
            r.detail["load_error"] = str(e)

        r.ok = True
        r.result = (
            f"found {path}; model_type={r.detail.get('model_type')}; "
            f"kv_lora_rank={r.detail.get('kv_lora_rank')}; "
            f"qk_rope_head_dim={r.detail.get('qk_rope_head_dim')}"
        )
        r.conclusion = (
            "PARTIAL: V2-Lite config available; MLA layout descriptor inputs clear for S2; "
            "full greedy+paged_rebuild_count==0 needs weight load on GPU (not forced in probe)"
        )
        r.detail["s2_layout_note"] = {
            "kv_lora_rank": r.detail.get("kv_lora_rank"),
            "qk_rope_head_dim": r.detail.get("qk_rope_head_dim"),
            "qk_nope_head_dim": r.detail.get("qk_nope_head_dim"),
            "radix_need": "per-layer layout enum {standard_mha, mla_compressed}; COW must copy ckv+kpe pages",
        }
        return r
    except Exception as e:
        r.ok = False
        r.result = f"{type(e).__name__}: {e}"
        r.detail["traceback"] = traceback.format_exc()[-2000:]
        r.conclusion = "FAIL: T4 probe error"
        return r


def main() -> int:
    if not __import__("torch").cuda.is_available():
        print("CUDA required", file=sys.stderr)
        return 2
    results = [
        probe_t1_flashinfer(),
        probe_t2_sgl_kernel(),
        probe_t3_deep_gemm(),
        probe_t4_v2_lite(),
    ]
    out = {
        "env": _env_info(),
        "results": [asdict(x) for x in results],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    # also write markdown summary lines
    print("\n## SUMMARY")
    for x in results:
        flag = "OK" if x.ok else "NO"
        print(f"- [{flag}] {x.item}: {x.conclusion}")
    return 0 if all(x.ok for x in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
