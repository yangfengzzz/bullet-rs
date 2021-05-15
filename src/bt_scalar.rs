use std::arch::x86_64::__m128;

pub type BtScalar = f32;
const BT_LARGE_FLOAT: f32 = 1e18;

pub type BtSimdFloat4 = __m128;

#[macro_export]
macro_rules! bt_castf_to128i {
    ($a:expr)=>{
        {
            $a as __m128i
        }
    }
}

#[macro_export]
macro_rules! bt_castf_to128d {
    ($a:expr)=>{
        {
            return $a as __m128d
        }
    }
}

#[macro_export]
macro_rules! bt_casti_to128f {
    ($a:expr)=>{
        {
            return $a as __m128
        }
    }
}

#[macro_export]
macro_rules! bt_castd_to128f {
    ($a:expr)=>{
        {
            return $a as __m128
        }
    }
}

#[macro_export]
macro_rules! bt_castd_to128i {
    ($a:expr)=>{
        {
            return $a as __m128i
        }
    }
}

#[macro_export]
macro_rules! bt_assign128 {
    ($r0:expr, $r1:expr, $r2:expr, $r3:expr)=>{
        {
            __m128($r0, $r1, $r2, $r3);
        }
    }
}