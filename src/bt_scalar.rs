/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use std::arch::x86_64::__m128;

pub type BtScalar = f32;

pub const BT_LARGE_FLOAT: f32 = 1e18;

pub type BtSimdFloat4 = __m128;

#[macro_export]
macro_rules! bt_castf_to128i {
    ($a:expr)=>{
        _mm_castps_si128($a)
    }
}

#[macro_export]
macro_rules! bt_castf_to128d {
    ($a:expr)=>{
        _mm_castps_pd($a)
    }
}

#[macro_export]
macro_rules! bt_casti_to128f {
    ($a:expr)=>{
        _mm_castsi128_ps($a)
    }
}

#[macro_export]
macro_rules! bt_castd_to128f {
    ($a:expr)=>{
        _mm_castpd_ps($a)
    }
}

#[macro_export]
macro_rules! bt_castd_to128i {
    ($a:expr)=>{
        _mm_castpd_si128($a)
    }
}

#[macro_export]
macro_rules! bt_assign128 {
    ($r0:expr, $r1:expr, $r2:expr, $r3:expr)=>{
        __m128($r0, $r1, $r2, $r3)
    }
}

pub const SIMD_EPSILON: f32 = f32::EPSILON;
pub const SIMD_INFINITY: f32 = f32::MAX;
pub const BT_ONE: f32 = 1.0;
pub const BT_ZERO: f32 = 0.0;
pub const BT_TWO: f32 = 2.0;
pub const BT_HALF: f32 = 0.5;