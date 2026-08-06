//! Minimal dtype conversions for safetensors payloads.

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
        other => Err(format!("unsupported safetensors dtype for f32 load: {other:?}")),
    }
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
}
