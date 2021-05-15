use crate::bt_scalar::BtScalar;

use std::arch::x86_64::{_mm_shuffle_ps, _mm_set_epi32};
#[macro_export]
macro_rules! bt_shuffle {
    ($x:expr,$y:expr, $z:expr, $w:expr)=>{
        {
            ($w << 6 | $z << 4 | $y << 2 | $x) & 0xff
        }
    }
}

#[macro_export]
macro_rules! bt_pshufd_ps {
    ($_a:expr,$_mask:expr)=>{
        {
            _mm_shuffle_ps($_a, $_a, $_mask)
        }
    }
}

#[macro_export]
macro_rules! bt_splat3_ps {
    ($_a:expr,$_i:expr)=>{
        {
            bt_pshufd_ps!($_a, bt_shuffle($_i, $_i, $_i, 3))
        }
    }
}

#[macro_export]
macro_rules! bt_splat_ps {
    ($_a:expr,$_i:expr)=>{
        {
            bt_pshufd_ps!($_a, bt_shuffle($_i, $_i, $_i, $_i))
        }
    }
}

#[macro_export]
macro_rules! btv_3absi_mask {
    ()=>{
        {
            _mm_set_epi32(0x00000000, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF)
        }
    }
}

#[macro_export]
macro_rules! btv_abs_mask {
    ()=>{
        {
            _mm_set_epi32(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF)
        }
    }
}

#[macro_export]
macro_rules! btv_fff0_mask {
    ()=>{
        {
            _mm_set_epi32(0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
        }
    }
}

#[macro_export]
pub fn test() {
    unsafe {
        let a = btv_3absi_mask!();
    }

    let b:BtScalar = 0.0;
}