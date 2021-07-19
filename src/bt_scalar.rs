/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use std::arch::x86_64::*;
use std::f32::consts::PI;

pub type BtScalar = f32;

pub const BT_LARGE_FLOAT: f32 = 1e18;

pub type BtSimdFloat4 = __m128;

pub fn bt_castf_to128i(a: BtSimdFloat4) -> __m128i {
    unsafe {
        return _mm_castps_si128(a);
    }
}

pub fn bt_castf_to128d(a: BtSimdFloat4) -> __m128d {
    unsafe {
        return _mm_castps_pd(a);
    }
}

pub fn bt_casti_to128f(a: __m128i) -> BtSimdFloat4 {
    unsafe {
        return _mm_castsi128_ps(a);
    }
}

pub fn bt_castd_to128f(a: __m128d) -> BtSimdFloat4 {
    unsafe {
        return _mm_castpd_ps(a);
    }
}

pub fn bt_castd_to128i(a: __m128d) -> __m128i {
    unsafe {
        return _mm_castpd_si128(a);
    }
}

pub fn bt_assign128(r0: f32, r1: f32, r2: f32, r3: f32) -> BtSimdFloat4 {
    unsafe {
        return _mm_set_ps(r0, r1, r2, r3);
    }
}

pub const SIMD_PI: BtScalar = std::f32::consts::PI;
pub const SIMD_2_PI: BtScalar = 2.0 * SIMD_PI;
pub const SIMD_HALF_PI: BtScalar = SIMD_PI * 0.5;
pub const SIMD_RADS_PER_DEG: BtScalar = SIMD_2_PI / 360.0;
pub const SIMD_DEGS_PER_RAD: BtScalar = 360.0 / SIMD_2_PI;
pub const SIMDSQRT12: BtScalar = std::f32::consts::FRAC_1_SQRT_2;

pub fn bt_recip_sqrt(x: BtScalar) -> BtScalar {
    return 1.0 / x.sqrt();
}

pub fn bt_recip(x: BtScalar) -> BtScalar {
    return x.recip();
}

pub const SIMD_EPSILON: f32 = f32::EPSILON;
pub const SIMD_INFINITY: f32 = f32::MAX;
pub const BT_ONE: f32 = 1.0;
pub const BT_ZERO: f32 = 0.0;
pub const BT_TWO: f32 = 2.0;
pub const BT_HALF: f32 = 0.5;

pub fn bt_atan2fast(y: BtScalar, x: BtScalar) -> BtScalar {
    let coeff_1 = PI / 4.0;
    let coeff_2 = 3.0 * coeff_1;
    let abs_y = y.abs();
    let angle;
    if x >= 0.0 {
        let r = (x - abs_y) / (x + abs_y);
        angle = coeff_1 - coeff_1 * r;
    } else {
        let r = (x + abs_y) / (abs_y - x);
        angle = coeff_2 - coeff_1 * r;
    }
    return if y < 0.0 { -angle } else { angle };
}

pub fn bt_fuzzy_zero(x: BtScalar) -> bool { return x.abs() < SIMD_EPSILON; }

pub fn bt_equal(a: BtScalar, eps: BtScalar) -> bool { return ((a) <= eps) && !((a) < -eps); }

pub fn bt_greater_equal(a: BtScalar, eps: BtScalar) -> bool { return !((a) <= eps); }

pub fn bt_is_negative(x: BtScalar) -> i32 { return if x < 0.0 { 1 } else { 0 }; }

pub fn bt_radians(x: BtScalar) -> BtScalar { return x * SIMD_RADS_PER_DEG; }

pub fn bt_degrees(x: BtScalar) -> BtScalar { return x * SIMD_DEGS_PER_RAD; }

pub fn bt_fsel(a: BtScalar, b: BtScalar, c: BtScalar) -> BtScalar { return if a >= 0.0 { b } else { c }; }

/// bt_select avoids branches, which makes performance much better for consoles like Playstation 3 and XBox 360
/// Thanks Phil Knight. See also http://www.cellperformance.com/articles/2006/04/more_techniques_for_eliminatin_1.html
pub fn bt_select_u32(condition: u32, value_if_condition_non_zero: u32,
                     value_if_condition_zero: u32) -> u32 {
    // Set test_nz to 0xFFFFFFFF if condition is nonzero, 0x00000000 if condition is zero
    // Rely on positive value or'ed with its negative having sign bit on
    // and zero value or'ed with its negative (which is still zero) having sign bit off
    // Use arithmetic shift right, shifting the sign bit through all 32 bits
    let test_nz = ((condition as i32 | -(condition as i32)) >> 31) as u32;
    let test_eqz = !test_nz;
    return (value_if_condition_non_zero & test_nz) | (value_if_condition_zero & test_eqz);
}

pub fn bt_select_i32(condition: u32, value_if_condition_non_zero: i32, value_if_condition_zero: i32) -> i32 {
    let test_nz = ((condition as i32 | -(condition as i32)) >> 31) as u32;
    let test_eqz = !test_nz;
    return (value_if_condition_non_zero & test_nz as i32) | (value_if_condition_zero & test_eqz as i32);
}

pub fn bt_select(condition: u32, value_if_condition_non_zero: f32, value_if_condition_zero: f32) -> f32 {
    return if condition != 0 { value_if_condition_non_zero } else { value_if_condition_zero };
}

pub fn bt_large_dot(mut a: &[BtScalar], mut b: &[BtScalar]) -> BtScalar {
    let mut p0: BtScalar;
    let mut q0: BtScalar;
    let mut m0: BtScalar;
    let mut p1: BtScalar;
    let mut q1: BtScalar;
    let mut m1: BtScalar;
    let mut sum = 0.0;
    let mut n = a.len() as isize;
    n -= 2;
    while n >= 0 {
        p0 = a[0];
        q0 = b[0];
        m0 = p0 * q0;
        p1 = a[1];
        q1 = b[1];
        m1 = p1 * q1;
        sum += m0;
        sum += m1;
        a = &a[2..];
        b = &b[2..];
        n -= 2;
    }
    while n > 0 {
        sum += a[0] * b[0];
        a = &a[1..];
        b = &b[1..];
        n -= 1;
    }
    return sum;
}

// returns normalized value in range [-SIMD_PI, SIMD_PI]
pub fn bt_normalize_angle(mut angle_in_radians: BtScalar) -> BtScalar {
    angle_in_radians = angle_in_radians % SIMD_2_PI;
    return if angle_in_radians < -SIMD_PI {
        angle_in_radians + SIMD_2_PI
    } else if angle_in_radians > SIMD_PI {
        angle_in_radians - SIMD_2_PI
    } else {
        angle_in_radians
    };
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
#[cfg(test)]
mod bt_scalar {
    use crate::bt_scalar::*;

    #[test]
    fn dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = bt_large_dot(&a, &b);
        assert_eq!(result, 1.0 + 4.0 + 9.0 + 16.0);
    }
}