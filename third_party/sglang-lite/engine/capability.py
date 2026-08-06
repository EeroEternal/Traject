"""Device capability probing and kernel routing tables (leaf selection only).

sglang-lite owns the routing decision; FlashInfer / sgl-kernel / DeepGEMM supply
leaf implementations. Never treat SM120 as SM100 — they are different families.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class ArchFamily(str, Enum):
    UNKNOWN = "unknown"
    SM80 = "sm80"
    SM90 = "sm90"
    SM100 = "sm100"
    SM120 = "sm120"


class SparseMlaBackend(str, Enum):
    NONE = "none"
    FLASHINFER_MLA = "flashinfer_mla"  # standard BatchMLA (non-sparse)
    FLASHINFER_SPARSE_SM120 = "flashinfer_sparse_sm120"
    FLASHINFER_SPARSE_SM100 = "flashinfer_sparse_sm100"
    OFFICIAL_SPARSE_ATTN = "official_sparse_attn"


class MoeGemmBackend(str, Enum):
    NONE = "none"
    TORCH = "torch"
    FLASHINFER_B12X = "flashinfer_b12x"
    SGL_KERNEL_FP8 = "sgl_kernel_fp8"
    DEEP_GEMM = "deep_gemm"


def capability_to_arch_family(major: int, minor: int) -> ArchFamily:
    """Map CUDA (major, minor) to an architecture family.

    Uses family buckets rather than ``major == 10`` checks so SM100 (10.x)
    and SM120 (12.x) never collapse into each other.
    """
    if major == 8:
        return ArchFamily.SM80
    if major == 9:
        return ArchFamily.SM90
    if major == 10:
        return ArchFamily.SM100
    if major == 12:
        return ArchFamily.SM120
    return ArchFamily.UNKNOWN


def probe_cuda_arch_family(device: Optional[int] = None) -> Tuple[ArchFamily, Tuple[int, int]]:
    """Return (arch_family, (major, minor)) for a CUDA device."""
    import torch

    if not torch.cuda.is_available():
        return ArchFamily.UNKNOWN, (0, 0)
    idx = 0 if device is None else int(device)
    major, minor = torch.cuda.get_device_capability(idx)
    return capability_to_arch_family(major, minor), (major, minor)


@dataclass(frozen=True)
class KernelCapabilities:
    """Declared leaf capabilities for the current process/device."""

    arch_family: ArchFamily
    cuda_capability: Tuple[int, int]
    flashinfer_version: Optional[str] = None
    has_sparse_mla_sm120: bool = False
    has_b12x_moe: bool = False
    has_sgl_kernel: bool = False
    has_deep_gemm_sm120: bool = False

    @property
    def sparse_mla_backend(self) -> SparseMlaBackend:
        return select_sparse_mla_backend(self)

    @property
    def moe_gemm_backend(self) -> MoeGemmBackend:
        return select_moe_gemm_backend(self)


def select_sparse_mla_backend(caps: KernelCapabilities) -> SparseMlaBackend:
    """Pick sparse MLA leaf. SM120 must never fall through to SM100 TRTLLM."""
    fam = caps.arch_family
    if fam == ArchFamily.SM120:
        if caps.has_sparse_mla_sm120:
            return SparseMlaBackend.FLASHINFER_SPARSE_SM120
        return SparseMlaBackend.OFFICIAL_SPARSE_ATTN
    if fam == ArchFamily.SM100:
        return SparseMlaBackend.FLASHINFER_SPARSE_SM100
    if fam in (ArchFamily.SM90, ArchFamily.SM80):
        return SparseMlaBackend.FLASHINFER_MLA
    return SparseMlaBackend.NONE


def select_moe_gemm_backend(caps: KernelCapabilities) -> MoeGemmBackend:
    """Prefer B12x / sgl-kernel on SM120; DeepGEMM only when probed OK."""
    if caps.arch_family == ArchFamily.SM120:
        if caps.has_deep_gemm_sm120:
            return MoeGemmBackend.DEEP_GEMM
        if caps.has_b12x_moe:
            return MoeGemmBackend.FLASHINFER_B12X
        if caps.has_sgl_kernel:
            return MoeGemmBackend.SGL_KERNEL_FP8
        return MoeGemmBackend.TORCH
    if caps.has_deep_gemm_sm120:
        return MoeGemmBackend.DEEP_GEMM
    if caps.has_sgl_kernel:
        return MoeGemmBackend.SGL_KERNEL_FP8
    if caps.has_b12x_moe:
        return MoeGemmBackend.FLASHINFER_B12X
    return MoeGemmBackend.TORCH


def probe_kernel_capabilities(device: str = "cuda") -> KernelCapabilities:
    """Runtime probe of arch family and available leaf modules."""
    arch = ArchFamily.UNKNOWN
    cap = (0, 0)
    fi_ver: Optional[str] = None
    has_sm120_sparse = False
    has_b12x = False
    has_sgl = False
    has_dg = False

    if device != "cpu":
        try:
            arch, cap = probe_cuda_arch_family()
        except Exception:
            pass

        try:
            import flashinfer

            fi_ver = getattr(flashinfer, "__version__", "?")
            try:
                import flashinfer.mla._sparse_mla_sm120  # noqa: F401

                has_sm120_sparse = True
            except Exception:
                has_sm120_sparse = False
            has_b12x = hasattr(flashinfer, "B12xMoEWrapper") or hasattr(
                flashinfer, "b12x_fused_moe"
            )
        except Exception:
            pass

        try:
            import sgl_kernel  # noqa: F401

            has_sgl = True
        except Exception:
            has_sgl = False

        # DeepGEMM SM120: only true after a successful tiny gemm probe.
        # Default False — deep_gemm 0.1.4 fails on sm_120.
        has_dg = False
        if arch != ArchFamily.SM120:
            try:
                import deep_gemm  # noqa: F401

                has_dg = True
            except Exception:
                has_dg = False

    return KernelCapabilities(
        arch_family=arch,
        cuda_capability=cap,
        flashinfer_version=fi_ver,
        has_sparse_mla_sm120=has_sm120_sparse,
        has_b12x_moe=has_b12x,
        has_sgl_kernel=has_sgl,
        has_deep_gemm_sm120=has_dg,
    )
