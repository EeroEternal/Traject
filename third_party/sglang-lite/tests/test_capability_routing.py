"""Unit tests: arch_family routing must never treat SM120 as SM100."""

from __future__ import annotations

from sglang_lite.capability import (
    ArchFamily,
    KernelCapabilities,
    MoeGemmBackend,
    SparseMlaBackend,
    capability_to_arch_family,
    select_moe_gemm_backend,
    select_sparse_mla_backend,
)


def test_capability_to_arch_family_buckets():
    assert capability_to_arch_family(8, 0) == ArchFamily.SM80
    assert capability_to_arch_family(9, 0) == ArchFamily.SM90
    assert capability_to_arch_family(10, 0) == ArchFamily.SM100
    assert capability_to_arch_family(12, 0) == ArchFamily.SM120
    assert capability_to_arch_family(7, 5) == ArchFamily.UNKNOWN


def test_sm120_never_selects_sm100_sparse_mla():
    caps = KernelCapabilities(
        arch_family=ArchFamily.SM120,
        cuda_capability=(12, 0),
        has_sparse_mla_sm120=False,
        has_b12x_moe=True,
    )
    assert select_sparse_mla_backend(caps) == SparseMlaBackend.OFFICIAL_SPARSE_ATTN
    assert select_sparse_mla_backend(caps) != SparseMlaBackend.FLASHINFER_SPARSE_SM100


def test_sm120_prefers_flashinfer_sparse_when_present():
    caps = KernelCapabilities(
        arch_family=ArchFamily.SM120,
        cuda_capability=(12, 0),
        has_sparse_mla_sm120=True,
    )
    assert select_sparse_mla_backend(caps) == SparseMlaBackend.FLASHINFER_SPARSE_SM120


def test_sm100_uses_sm100_sparse():
    caps = KernelCapabilities(
        arch_family=ArchFamily.SM100,
        cuda_capability=(10, 0),
        has_sparse_mla_sm120=False,
    )
    assert select_sparse_mla_backend(caps) == SparseMlaBackend.FLASHINFER_SPARSE_SM100


def test_sm120_moe_prefers_b12x_over_deepgemm_default():
    caps = KernelCapabilities(
        arch_family=ArchFamily.SM120,
        cuda_capability=(12, 0),
        has_b12x_moe=True,
        has_sgl_kernel=True,
        has_deep_gemm_sm120=False,
    )
    assert select_moe_gemm_backend(caps) == MoeGemmBackend.FLASHINFER_B12X


def test_sm120_moe_deepgemm_only_when_probed():
    caps = KernelCapabilities(
        arch_family=ArchFamily.SM120,
        cuda_capability=(12, 0),
        has_b12x_moe=True,
        has_deep_gemm_sm120=True,
    )
    assert select_moe_gemm_backend(caps) == MoeGemmBackend.DEEP_GEMM


def test_kernel_capabilities_properties():
    caps = KernelCapabilities(
        arch_family=ArchFamily.SM120,
        cuda_capability=(12, 0),
        has_sparse_mla_sm120=True,
        has_b12x_moe=True,
    )
    assert caps.sparse_mla_backend == SparseMlaBackend.FLASHINFER_SPARSE_SM120
    assert caps.moe_gemm_backend == MoeGemmBackend.FLASHINFER_B12X
