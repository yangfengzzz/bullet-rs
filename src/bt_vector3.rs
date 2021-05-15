use crate::bt_scalar::BtScalar;

use std::arch::x86_64::{_mm_shuffle_ps, _mm_set_epi32, __m128i};
#[macro_export]
macro_rules! bt_shuffle {
    ($x:expr,$y:expr, $z:expr, $w:expr)=>{
        ($w << 6 | $z << 4 | $y << 2 | $x) & 0xff
    }
}

#[macro_export]
macro_rules! bt_pshufd_ps {
    ($_a:expr,$_mask:expr)=>{
        unsafe {
            _mm_shuffle_ps($_a, $_a, $_mask)
        }
    }
}

#[macro_export]
macro_rules! bt_splat3_ps {
    ($_a:expr,$_i:expr)=>{
        bt_pshufd_ps!($_a, bt_shuffle($_i, $_i, $_i, 3))
    }
}

#[macro_export]
macro_rules! bt_splat_ps {
    ($_a:expr,$_i:expr)=>{
            bt_pshufd_ps!($_a, bt_shuffle($_i, $_i, $_i, $_i))
    }
}

#[macro_export]
macro_rules! btv_3absi_mask {
    ()=>{
        unsafe {
            _mm_set_epi32(0x00000000, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF)
        }
    }
}

#[macro_export]
macro_rules! btv_abs_mask {
    ()=>{
       unsafe {
            _mm_set_epi32(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF)
        }
    }
}

#[macro_export]
macro_rules! btv_fff0_mask {
    ()=>{
        unsafe {
            _mm_set_epi32(0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
        }
    }
}

#[macro_export]
macro_rules! btv3_absf_mask {
    () => {
        bt_casti_to128f!(btv_3absi_mask!())
    };
}

#[macro_export]
macro_rules! btv_fff0f_mask {
    () => {
        bt_casti_to128f!(btv_fff0_mask!())
    };
}

#[macro_export]
macro_rules! btv_mzero_mask {
    () => {
        unsafe {
            _mm_set_ps(-0.0, -0.0, -0.0, -0.0)
        }
    };
}

#[macro_export]
macro_rules! v_1110 {
    () => {
        unsafe {
            _mm_set_ps(0.0, 1.0, 1.0, 1.0)
        }
    };
}

#[macro_export]
macro_rules! v_half {
    () => {
        unsafe {
            _mm_set_ps(0.5, 0.5, 0.5, 0.5)
        }
    };
}

#[macro_export]
macro_rules! v_1_5 {
    () => {
        unsafe {
            _mm_set_ps(1.5, 1.5, 1.5, 1.5)
        }
    };
}

#[macro_export]
pub fn test() {
    let a = btv_3absi_mask!();

    let b: BtScalar = 0.0;
}