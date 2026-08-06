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
}
