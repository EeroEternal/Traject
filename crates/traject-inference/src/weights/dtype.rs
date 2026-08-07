//! Minimal dtype conversions for safetensors payloads (BF16/F16/F32/FP8).

/// IEEE BF16 bits → f32 (shift into high half of f32).
#[inline]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// IEEE F16 bits → f32.
#[inline]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    // Portable half→float without `half` crate.
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x3ff;
    let f = if exp == 0 {
        if frac == 0 {
            0.0
        } else {
            // subnormal
            let mut f = frac as f32 / 1024.0;
            f *= 2f32.powi(-14);
            f
        }
    } else if exp == 31 {
        if frac == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        let mut f = 1.0 + (frac as f32) / 1024.0;
        f *= 2f32.powi(exp as i32 - 15);
        f
    };
    if sign == 1 {
        -f
    } else {
        f
    }
}

/// OCP FP8 E4M3 (1 sign / 4 exp / 3 mant, bias 7) → f32.
#[inline]
pub fn e4m3_bits_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0x0f;
    let mant = bits & 0x07;
    let val = if exp == 0 {
        if mant == 0 {
            0.0
        } else {
            (mant as f32 / 8.0) * 2f32.powi(-6)
        }
    } else if exp == 0x0f && mant == 0x07 {
        f32::NAN
    } else {
        (1.0 + mant as f32 / 8.0) * 2f32.powi(exp as i32 - 7)
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// UE8M0 / F8_E8M0 power-of-two scale: `2^(byte - 127)`.
#[inline]
pub fn e8m0_bits_to_f32(bits: u8) -> f32 {
    2f32.powi(bits as i32 - 127)
}

/// OCP FP4 E2M1 nibble (1 sign / 2 exp / 1 mant) → f32.
///
/// Values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}.
#[inline]
pub fn e2m1_nibble_to_f32(nibble: u8) -> f32 {
    let n = nibble & 0x0f;
    let sign = (n >> 3) & 1;
    let exp = (n >> 1) & 0x03;
    let mant = n & 1;
    let val = if exp == 0 {
        0.5 * mant as f32
    } else {
        (1.0 + 0.5 * mant as f32) * 2f32.powi(exp as i32 - 1)
    };
    if sign == 1 {
        -val
    } else {
        val
    }
}

/// Dequant packed FP4 (two e2m1 per byte along K) with per-row e8m0 block scales.
///
/// DeepSeek-V4 experts: packed `I8` weight `[rows, cols/2]`, scale `F8_E8M0`
/// `[rows, cols/block_k]`, `block_k=32`. Low nibble is even column.
pub fn dequant_fp4_block_scaled(
    packed: &[u8],
    packed_shape: &[usize],
    scale_e8m0: &[u8],
    scale_shape: &[usize],
    block_k: usize,
) -> Result<Vec<f32>, String> {
    if packed_shape.len() != 2 || scale_shape.len() != 2 {
        return Err(format!(
            "fp4 dequant expects 2D, packed={packed_shape:?} scale={scale_shape:?}"
        ));
    }
    let rows = packed_shape[0];
    let packed_cols = packed_shape[1];
    let cols = packed_cols.saturating_mul(2);
    let sr = scale_shape[0];
    let sc = scale_shape[1];
    let block_k = block_k.max(1);
    if sr < rows || sc * block_k < cols {
        return Err(format!(
            "fp4 scale {scale_shape:?} * block_k {block_k} too small for logical [{rows}, {cols}]"
        ));
    }
    if packed.len() != rows * packed_cols {
        return Err(format!(
            "packed len {} != rows*packed_cols {}",
            packed.len(),
            rows * packed_cols
        ));
    }
    if scale_e8m0.len() != sr * sc {
        return Err(format!(
            "scale len {} != sr*sc {}",
            scale_e8m0.len(),
            sr * sc
        ));
    }

    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for jb in 0..packed_cols {
            let b = packed[i * packed_cols + jb];
            for nibble in 0..2 {
                let j = jb * 2 + nibble;
                let bits = if nibble == 0 { b & 0x0f } else { b >> 4 };
                let w = e2m1_nibble_to_f32(bits);
                let sj = j / block_k;
                let s = e8m0_bits_to_f32(scale_e8m0[i * sc + sj]);
                out[i * cols + j] = w * s;
            }
        }
    }
    Ok(out)
}

/// Fused `y = W x` for packed FP4 row-major weights (no full dequant buffer).
///
/// `x.len() >= packed_cols * 2`, `y.len() == rows`.
pub fn matvec_fp4_block_scaled(
    packed: &[u8],
    rows: usize,
    packed_cols: usize,
    scale_e8m0: &[u8],
    scale_cols: usize,
    block_k: usize,
    x: &[f32],
) -> Result<Vec<f32>, String> {
    let cols = packed_cols.saturating_mul(2);
    let block_k = block_k.max(1);
    if x.len() < cols {
        return Err(format!("fp4 matvec x len {} < cols {cols}", x.len()));
    }
    if packed.len() != rows * packed_cols {
        return Err(format!(
            "fp4 matvec packed len {} != rows*packed_cols {}",
            packed.len(),
            rows * packed_cols
        ));
    }
    if scale_e8m0.len() < rows * scale_cols {
        return Err(format!(
            "fp4 matvec scale len {} < rows*scale_cols {}",
            scale_e8m0.len(),
            rows * scale_cols
        ));
    }
    let mut y = vec![0.0f32; rows];
    for i in 0..rows {
        let mut acc = 0.0f32;
        let row_off = i * packed_cols;
        let scale_row = i * scale_cols;
        for jb in 0..packed_cols {
            let b = packed[row_off + jb];
            let j0 = jb * 2;
            let j1 = j0 + 1;
            let s0 = e8m0_bits_to_f32(scale_e8m0[scale_row + j0 / block_k]);
            let s1 = e8m0_bits_to_f32(scale_e8m0[scale_row + j1 / block_k]);
            let w0 = e2m1_nibble_to_f32(b & 0x0f) * s0;
            let w1 = e2m1_nibble_to_f32(b >> 4) * s1;
            acc += w0 * x[j0] + w1 * x[j1];
        }
        y[i] = acc;
    }
    Ok(y)
}

pub fn bytes_to_f32_vec(data: &[u8], dtype: safetensors::Dtype) -> Result<Vec<f32>, String> {
    match dtype {
        safetensors::Dtype::F32 => {
            if data.len() % 4 != 0 {
                return Err("F32 tensor byte length not multiple of 4".into());
            }
            let mut out = Vec::with_capacity(data.len() / 4);
            for chunk in data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        safetensors::Dtype::BF16 => {
            if data.len() % 2 != 0 {
                return Err("BF16 tensor byte length not multiple of 2".into());
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(bf16_bits_to_f32(bits));
            }
            Ok(out)
        }
        safetensors::Dtype::F16 => {
            if data.len() % 2 != 0 {
                return Err("F16 tensor byte length not multiple of 2".into());
            }
            let mut out = Vec::with_capacity(data.len() / 2);
            for chunk in data.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f16_bits_to_f32(bits));
            }
            Ok(out)
        }
        safetensors::Dtype::F8_E4M3 => {
            let mut out = Vec::with_capacity(data.len());
            for &b in data {
                out.push(e4m3_bits_to_f32(b));
            }
            Ok(out)
        }
        safetensors::Dtype::F8_E8M0 => {
            let mut out = Vec::with_capacity(data.len());
            for &b in data {
                out.push(e8m0_bits_to_f32(b));
            }
            Ok(out)
        }
        other => Err(format!("unsupported safetensors dtype for f32 load: {other:?}")),
    }
}

/// Fused matvec for block-scaled FP8: `y = (e4m3 * e8m0_scale) @ x` without full dequant.
///
/// Weight `[rows, cols]` e4m3, scale `[ceil(rows/B), ceil(cols/B)]` e8m0, block `B` (V4: 128).
pub fn matvec_fp8_block_scaled(
    weight_e4m3: &[u8],
    rows: usize,
    cols: usize,
    scale_e8m0: &[u8],
    scale_rows: usize,
    scale_cols: usize,
    block: usize,
    x: &[f32],
) -> Result<Vec<f32>, String> {
    let block = block.max(1);
    if x.len() < cols {
        return Err(format!("fp8 matvec x len {} < cols {cols}", x.len()));
    }
    if weight_e4m3.len() != rows * cols {
        return Err(format!(
            "fp8 matvec weight len {} != rows*cols {}",
            weight_e4m3.len(),
            rows * cols
        ));
    }
    if scale_e8m0.len() < scale_rows * scale_cols {
        return Err(format!(
            "fp8 matvec scale len {} < scale_rows*scale_cols {}",
            scale_e8m0.len(),
            scale_rows * scale_cols
        ));
    }
    let mut y = vec![0.0f32; rows];
    for i in 0..rows {
        let si = i / block;
        let mut acc = 0.0f32;
        let row_off = i * cols;
        for j in 0..cols {
            let sj = j / block;
            let w = e4m3_bits_to_f32(weight_e4m3[row_off + j]);
            let s = e8m0_bits_to_f32(scale_e8m0[si * scale_cols + sj]);
            acc += w * s * x[j];
        }
        y[i] = acc;
    }
    Ok(y)
}

/// Dot product of one FP8 row with `x` (same scaling as [`matvec_fp8_block_scaled`]).
pub fn row_dot_fp8_block_scaled(
    weight_e4m3: &[u8],
    rows: usize,
    cols: usize,
    scale_e8m0: &[u8],
    scale_cols: usize,
    block: usize,
    row: usize,
    x: &[f32],
) -> Result<f32, String> {
    if row >= rows {
        return Err(format!("fp8 row_dot row {row} >= rows {rows}"));
    }
    if x.len() < cols {
        return Err(format!("fp8 row_dot x len {} < cols {cols}", x.len()));
    }
    let block = block.max(1);
    let si = row / block;
    let row_off = row * cols;
    let mut acc = 0.0f32;
    for j in 0..cols {
        let sj = j / block;
        let w = e4m3_bits_to_f32(weight_e4m3[row_off + j]);
        let s = e8m0_bits_to_f32(scale_e8m0[si * scale_cols + sj]);
        acc += w * s * x[j];
    }
    Ok(acc)
}

/// In-place normalized Walsh–Hadamard transform (FWHT).
///
/// Matches `fast_hadamard_transform.hadamard_transform(x, scale=dim**-0.5)` when
/// `scale = (len as f32).sqrt().recip()`. `x.len()` must be a power of two.
pub fn hadamard_transform_inplace(x: &mut [f32], scale: f32) {
    let n = x.len();
    if n == 0 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "hadamard len {n} must be power of two");
    let mut h = 1usize;
    while h < n {
        let step = h * 2;
        for i in (0..n).step_by(step) {
            for j in 0..h {
                let a = x[i + j];
                let b = x[i + j + h];
                x[i + j] = a + b;
                x[i + j + h] = a - b;
            }
        }
        h = step;
    }
    if (scale - 1.0).abs() > f32::EPSILON {
        for v in x.iter_mut() {
            *v *= scale;
        }
    }
}

/// Official `rotate_activation`: Hadamard with `scale = dim ** -0.5`.
///
/// No-op when `x.len()` is not a power of two (production index heads use 128).
pub fn rotate_activation_inplace(x: &mut [f32]) {
    let n = x.len();
    if n == 0 || !n.is_power_of_two() {
        return;
    }
    let scale = (n as f32).sqrt().recip();
    hadamard_transform_inplace(x, scale);
}

/// Nearest OCP FP4 E2M1 codepoint for a scalar in `[-6, 6]`.
#[inline]
pub fn f32_to_nearest_e2m1(v: f32) -> f32 {
    // Positive codebook: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    const POS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = if v.is_sign_negative() { -1.0 } else { 1.0 };
    let a = v.abs().min(6.0);
    let mut best = POS[0];
    let mut best_d = (a - best).abs();
    for &c in &POS[1..] {
        let d = (a - c).abs();
        if d < best_d {
            best = c;
            best_d = d;
        }
    }
    sign * best
}

/// `ceil(log2(x))` via IEEE-754 bit ops (matches TileLang `fast_log2_ceil`).
#[inline]
fn fast_log2_ceil(x: f32) -> i32 {
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    exp - 127 + if man != 0 { 1 } else { 0 }
}

/// Power-of-two scale for FP4 QAT: `2^ceil(log2(amax / 6))`.
#[inline]
fn fp4_round_scale(amax: f32) -> f32 {
    // Floor amax so scale stays in e8m0 range (official: max(amax, 6 * 2^-126)).
    let amax = amax.max(6.0 * 2f32.powi(-126));
    let t = amax / 6.0;
    2f32.powi(fast_log2_ceil(t))
}

/// Block-wise FP4 quant → dequant (QAT simulation), matching `fp4_act_quant(..., inplace=True)`.
///
/// Groups along the last axis of length `x.len()` (one row). When `len % block_size != 0`,
/// falls back to a single block of full length (unit tests with tiny dims).
pub fn fp4_act_quant_inplace(x: &mut [f32], block_size: usize) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let bs = if block_size > 0 && n % block_size == 0 {
        block_size
    } else {
        n
    };
    let mut i = 0;
    while i < n {
        let end = (i + bs).min(n);
        let mut amax = 0.0f32;
        for v in &x[i..end] {
            amax = amax.max(v.abs());
        }
        let s = fp4_round_scale(amax);
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        for v in &mut x[i..end] {
            let q = f32_to_nearest_e2m1(*v * inv);
            *v = q * s;
        }
        i = end;
    }
}

/// Indexer Q/K path: Hadamard rotate then FP4 QAT sim (official `Indexer.forward`).
pub fn indexer_qk_qat_inplace(x: &mut [f32], fp4_block: usize) {
    rotate_activation_inplace(x);
    fp4_act_quant_inplace(x, fp4_block);
}

/// Block-scaled FP8 dequant: `weight[out,in]` with `scale[out/B, in/B]`, block size `B`.
///
/// DeepSeek-V4 uses B=128: e4m3 weights × e8m0 scales.
pub fn dequant_fp8_block_scaled(
    weight_e4m3: &[u8],
    weight_shape: &[usize],
    scale_e8m0: &[u8],
    scale_shape: &[usize],
    block: usize,
) -> Result<Vec<f32>, String> {
    if weight_shape.len() != 2 || scale_shape.len() != 2 {
        return Err(format!(
            "fp8 block dequant expects 2D weight/scale, got {weight_shape:?} / {scale_shape:?}"
        ));
    }
    let rows = weight_shape[0];
    let cols = weight_shape[1];
    let sr = scale_shape[0];
    let sc = scale_shape[1];
    let block = block.max(1);
    if sr * block < rows || sc * block < cols {
        return Err(format!(
            "scale shape {scale_shape:?} * block {block} too small for weight {weight_shape:?}"
        ));
    }
    if weight_e4m3.len() != rows * cols {
        return Err(format!(
            "weight bytes {} != rows*cols {}",
            weight_e4m3.len(),
            rows * cols
        ));
    }
    if scale_e8m0.len() != sr * sc {
        return Err(format!(
            "scale bytes {} != sr*sc {}",
            scale_e8m0.len(),
            sr * sc
        ));
    }

    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        let si = i / block;
        for j in 0..cols {
            let sj = j / block;
            let w = e4m3_bits_to_f32(weight_e4m3[i * cols + j]);
            let s = e8m0_bits_to_f32(scale_e8m0[si * sc + sj]);
            out[i * cols + j] = w * s;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_one() {
        // 1.0 in bf16 is 0x3f80
        let v = bf16_bits_to_f32(0x3f80);
        assert!((v - 1.0).abs() < 1e-5, "v={v}");
    }

    #[test]
    fn e4m3_known() {
        // byte 65 → 2.25 (validated vs torch.float8_e4m3fn)
        assert!((e4m3_bits_to_f32(65) - 2.25).abs() < 1e-5);
        assert!((e4m3_bits_to_f32(224) + 32.0).abs() < 1e-5);
    }

    #[test]
    fn e8m0_known() {
        // byte 115 → 2^(115-127) = 2^-12 = 0.000244140625
        assert!((e8m0_bits_to_f32(115) - 0.000244140625).abs() < 1e-12);
        assert!((e8m0_bits_to_f32(116) - 0.00048828125).abs() < 1e-12);
    }

    #[test]
    fn block_dequant_tiny() {
        // 2x2 weight, block=2 → scale 1x1
        let w = [65u8, 65, 65, 65]; // 2.25 each
        let s = [127u8]; // 2^0 = 1.0
        let out = dequant_fp8_block_scaled(&w, &[2, 2], &s, &[1, 1], 2).unwrap();
        assert_eq!(out.len(), 4);
        for v in out {
            assert!((v - 2.25).abs() < 1e-5, "v={v}");
        }
    }

    #[test]
    fn fp8_matvec_matches_dequant() {
        // 2x2, block=2, all weights 2.25, scale 1.0
        let w = [65u8, 65, 65, 65];
        let s = [127u8];
        let x = [1.0f32, 2.0];
        let deq = dequant_fp8_block_scaled(&w, &[2, 2], &s, &[1, 1], 2).unwrap();
        let y_ref = [
            deq[0] * x[0] + deq[1] * x[1],
            deq[2] * x[0] + deq[3] * x[1],
        ];
        let y = matvec_fp8_block_scaled(&w, 2, 2, &s, 1, 1, 2, &x).unwrap();
        for (a, b) in y.iter().zip(y_ref.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        let d0 = row_dot_fp8_block_scaled(&w, 2, 2, &s, 1, 2, 0, &x).unwrap();
        assert!((d0 - y_ref[0]).abs() < 1e-5);
    }

    #[test]
    fn e2m1_known() {
        assert!((e2m1_nibble_to_f32(0) - 0.0).abs() < 1e-6);
        assert!((e2m1_nibble_to_f32(1) - 0.5).abs() < 1e-6);
        assert!((e2m1_nibble_to_f32(2) - 1.0).abs() < 1e-6);
        assert!((e2m1_nibble_to_f32(7) - 6.0).abs() < 1e-6);
        assert!((e2m1_nibble_to_f32(10) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn fp4_block_dequant_tiny() {
        // 1 row, 2 logical cols, packed 1 byte; scale block_k=2 → 1 scale
        // nibble0=2 → 1.0, nibble1=2 → 1.0; scale 127 → 1.0
        let packed = [2u8 | (2u8 << 4)];
        let scale = [127u8];
        let out = dequant_fp4_block_scaled(&packed, &[1, 1], &scale, &[1, 1], 2).unwrap();
        assert_eq!(out.len(), 2);
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fp4_matvec_matches_dequant() {
        // 2 rows × 4 cols (2 packed cols), block_k=2, scale [2, 2]
        let packed = [
            2u8 | (2 << 4), // row0: 1,1
            2u8 | (2 << 4),
            4u8 | (0 << 4), // row1: 2,0
            1u8 | (2 << 4), // 0.5, 1
        ];
        let scale = [127u8, 127, 127, 127]; // all 1.0
        let w = dequant_fp4_block_scaled(&packed, &[2, 2], &scale, &[2, 2], 2).unwrap();
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y_ref = {
            let mut y = vec![0.0f32; 2];
            for i in 0..2 {
                for j in 0..4 {
                    y[i] += w[i * 4 + j] * x[j];
                }
            }
            y
        };
        let y = matvec_fp4_block_scaled(&packed, 2, 2, &scale, 2, 2, &x).unwrap();
        for (a, b) in y.iter().zip(y_ref.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn hadamard_dim2_normalized() {
        // H2 * [1,0] / sqrt(2) = [1,1]/sqrt(2)
        let mut x = [1.0f32, 0.0];
        rotate_activation_inplace(&mut x);
        let s = 2f32.sqrt().recip();
        assert!((x[0] - s).abs() < 1e-5, "x0={}", x[0]);
        assert!((x[1] - s).abs() < 1e-5, "x1={}", x[1]);
    }

    #[test]
    fn hadamard_involutive_up_to_sign_scale() {
        // Normalized FWHT is involutive: H(H(x)) = x
        let mut x = [1.0f32, 2.0, 3.0, 4.0];
        let orig = x;
        rotate_activation_inplace(&mut x);
        rotate_activation_inplace(&mut x);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn e2m1_nearest_known() {
        assert!((f32_to_nearest_e2m1(0.1) - 0.0).abs() < 1e-6);
        assert!((f32_to_nearest_e2m1(0.4) - 0.5).abs() < 1e-6);
        assert!((f32_to_nearest_e2m1(5.0) - 4.0).abs() < 1e-6 || (f32_to_nearest_e2m1(5.0) - 6.0).abs() < 1e-6);
        assert!((f32_to_nearest_e2m1(-1.2) + 1.0).abs() < 1e-6 || (f32_to_nearest_e2m1(-1.2) + 1.5).abs() < 1e-6);
        assert!((f32_to_nearest_e2m1(10.0) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn fp4_act_quant_roundtrips_exact_codebook() {
        // Values already on the grid with amax=6 → scale=1
        let mut x = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let orig = x;
        fp4_act_quant_inplace(&mut x, 8);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn fp4_act_quant_scales_large_values() {
        // amax=12 → scale = 2 (ceil log2(12/6)=ceil log2(2)=1 → 2)
        let mut x = [12.0f32; 32];
        fp4_act_quant_inplace(&mut x, 32);
        // 12/2=6 → e2m1 6 → *2 = 12
        for v in &x {
            assert!((v - 12.0).abs() < 1e-4, "v={v}");
        }
    }

    #[test]
    fn indexer_qk_qat_finite() {
        let mut x: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 - 0.5).collect();
        indexer_qk_qat_inplace(&mut x, 32);
        assert!(x.iter().all(|v| v.is_finite()));
    }
}
