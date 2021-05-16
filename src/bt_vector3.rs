use crate::bt_scalar::{*};
use crate::bt_casti_to128f;
use std::arch::x86_64::{*};
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign, Add, Mul, Sub, Neg, Div};

#[macro_export]
macro_rules! bt_shuffle {
    ($x:expr,$y:expr, $z:expr, $w:expr)=>{
        ($w << 6 | $z << 4 | $y << 2 | $x) & 0xff
    }
}

#[macro_export]
macro_rules! bt_pshufd_ps {
    ($_a:expr,$_mask:expr)=>{
        _mm_shuffle_ps($_a, $_a, $_mask)
    }
}

#[macro_export]
macro_rules! bt_splat3_ps {
    ($_a:expr,$_i:expr)=>{
        bt_pshufd_ps!($_a, bt_shuffle!($_i, $_i, $_i, 3))
    }
}

#[macro_export]
macro_rules! bt_splat_ps {
    ($_a:expr,$_i:expr)=>{
        bt_pshufd_ps!($_a, bt_shuffle!($_i, $_i, $_i, $_i))
    }
}

#[macro_export]
macro_rules! btv_3absi_mask {
    ()=>{
        _mm_set_epi32(0x00000000, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF)
    }
}

#[macro_export]
macro_rules! btv_abs_mask {
    ()=>{
        _mm_set_epi32(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF)
    }
}

#[macro_export]
macro_rules! btv_fff0_mask {
    ()=>{
        _mm_set_epi32(0x00000000, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)
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
        _mm_set_ps(-0.0, -0.0, -0.0, -0.0)
    };
}

#[macro_export]
macro_rules! v_1110 {
    () => {
        _mm_set_ps(0.0, 1.0, 1.0, 1.0)
    };
}

#[macro_export]
macro_rules! v_half {
    () => {
        _mm_set_ps(0.5, 0.5, 0.5, 0.5)
    };
}

#[macro_export]
macro_rules! v_1_5 {
    () => {
        _mm_set_ps(1.5, 1.5, 1.5, 1.5)
    };
}

//--------------------------------------------------------------------------------------------------
union SimdToArray {
    array: [f32; 4],
    simd: BtSimdFloat4,
}

pub struct BtVector3 {
    m_vec128: SimdToArray,
}

impl BtVector3 {
    #[inline(always)]
    pub fn get128(&self) -> BtSimdFloat4 { unsafe { return self.m_vec128.simd; } }

    #[inline(always)]
    pub fn set128(&mut self, v128: BtSimdFloat4) { self.m_vec128.simd = v128; }

    pub fn new_default() -> BtVector3 {
        return BtVector3 {
            m_vec128: SimdToArray { array: [0.0, 0.0, 0.0, 0.0] }
        };
    }

    /// Constructor from scalars
    pub fn new(_x: BtScalar, _y: BtScalar, _z: BtScalar) -> BtVector3 {
        return BtVector3 {
            m_vec128: SimdToArray { array: [_x, _y, _z, 0.0] }
        };
    }

    /// Set Vector
    pub fn new_simd(v: BtSimdFloat4) -> BtVector3 {
        return BtVector3 {
            m_vec128: SimdToArray { simd: v }
        };
    }
}

impl AddAssign for BtVector3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.m_vec128.simd = unsafe { _mm_sub_ps(self.m_vec128.simd, rhs.m_vec128.simd) }
    }
}

impl SubAssign for BtVector3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.m_vec128.simd = unsafe { _mm_sub_ps(self.m_vec128.simd, rhs.m_vec128.simd) }
    }
}

impl MulAssign<BtScalar> for BtVector3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: BtScalar) {
        unsafe {
            let mut vs = _mm_load_ss(&rhs); //    (S 0 0 0)
            vs = bt_pshufd_ps!(vs, 0x80); //    (S S S 0.0)
            self.m_vec128.simd = _mm_mul_ps(self.m_vec128.simd, vs)
        }
    }
}

impl MulAssign<&BtScalar> for BtVector3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &BtScalar) {
        unsafe {
            let mut vs = _mm_load_ss(&*rhs); //    (S 0 0 0)
            vs = bt_pshufd_ps!(vs, 0x80); //    (S S S 0.0)
            self.m_vec128.simd = _mm_mul_ps(self.m_vec128.simd, vs)
        }
    }
}

impl DivAssign<BtScalar> for BtVector3 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: BtScalar) {
        self.mul_assign(1.0 / rhs)
    }
}

macro_rules! impl_bin_add_vector {
    ($lhs:ty, $rhs:ty) => {
        /// Return the sum of two vectors (Point symantics)
        impl Add<$rhs> for $lhs {
            type Output = BtVector3;

            fn add(self, rhs: $rhs) -> Self::Output {
                unsafe {
                    return BtVector3::new_simd(_mm_add_ps(self.m_vec128.simd, rhs.m_vec128.simd));
                }
            }
        }
    }
}

impl_bin_add_vector!(BtVector3, BtVector3);
impl_bin_add_vector!(&BtVector3, BtVector3);
impl_bin_add_vector!(BtVector3, &BtVector3);
impl_bin_add_vector!(&BtVector3, &BtVector3);

macro_rules! impl_bin_mul_vector {
    ($lhs:ty, $rhs:ty) => {
        /// Return the sum of two vectors (Point symantics)
        impl Mul<$rhs> for $lhs {
            type Output = BtVector3;

            fn mul(self, rhs: $rhs) -> Self::Output {
                unsafe {
                    return BtVector3::new_simd(_mm_mul_ps(self.m_vec128.simd, rhs.m_vec128.simd));
                }
            }
        }
    }
}

impl_bin_mul_vector!(BtVector3, BtVector3);
impl_bin_mul_vector!(&BtVector3, BtVector3);
impl_bin_mul_vector!(BtVector3, &BtVector3);
impl_bin_mul_vector!(&BtVector3, &BtVector3);

macro_rules! impl_bin_sub_vector {
    ($lhs:ty, $rhs:ty) => {
        // Return the difference between two vectors
        impl Sub<$rhs> for $lhs {
            type Output = BtVector3;

            #[allow(overflowing_literals)]
            fn sub(self, rhs: $rhs) -> Self::Output {
                unsafe {
                    let r = _mm_sub_ps(self.m_vec128.simd, rhs.m_vec128.simd);
                    let b = btv_fff0f_mask!();
                    return BtVector3::new_simd(_mm_and_ps(r, b));
                }
            }
        }
    }
}

impl_bin_sub_vector!(BtVector3, BtVector3);
impl_bin_sub_vector!(&BtVector3, BtVector3);
impl_bin_sub_vector!(BtVector3, &BtVector3);
impl_bin_sub_vector!(&BtVector3, &BtVector3);

/// Return the negative of the vector
impl Neg for BtVector3 {
    type Output = BtVector3;

    #[allow(overflowing_literals)]
    fn neg(self) -> Self::Output {
        unsafe {
            let r = _mm_xor_ps(self.m_vec128.simd, btv_mzero_mask!());
            return BtVector3::new_simd(_mm_and_ps(r, btv_fff0f_mask!()));
        }
    }
}

impl Mul<BtScalar> for BtVector3 {
    type Output = BtVector3;

    fn mul(self, rhs: f32) -> Self::Output {
        unsafe {
            let mut vs = _mm_load_ss(&rhs); //    (S 0 0 0)
            vs = bt_pshufd_ps!(vs, 0x80); //    (S S S 0.0)
            return BtVector3::new_simd(_mm_mul_ps(self.m_vec128.simd, vs));
        }
    }
}

impl Div<BtScalar> for BtVector3 {
    type Output = BtVector3;

    fn div(self, rhs: f32) -> Self::Output {
        return self * 1.0 / rhs;
    }
}

impl Div for BtVector3 {
    type Output = BtVector3;

    #[allow(overflowing_literals)]
    fn div(self, rhs: Self) -> Self::Output {
        unsafe {
            let mut vec = _mm_div_ps(self.m_vec128.simd, rhs.m_vec128.simd);
            vec = _mm_and_ps(vec, btv_fff0f_mask!());
            return BtVector3::new_simd(vec);
        }
    }
}

//--------------------------------------------------------------------------------------------------
impl BtVector3 {
    #[inline(always)]
    pub fn dot(&self, v: &BtVector3) -> BtScalar {
        unsafe {
            let mut vd = _mm_mul_ps(self.m_vec128.simd, v.m_vec128.simd);
            let z = _mm_movehl_ps(vd, vd);
            let y = _mm_shuffle_ps(vd, vd, 0x55);
            vd = _mm_add_ss(vd, y);
            vd = _mm_add_ss(vd, z);
            return _mm_cvtss_f32(vd);
        }
    }

    /// Return the length of the vector squared
    #[inline(always)]
    pub fn length2(&self) -> BtScalar { return self.dot(self); }

    /// Return the length of the vector
    #[inline(always)]
    pub fn length(&self) -> BtScalar {
        return self.length2().sqrt();
    }

    /// Return the norm (length) of the vector
    #[inline(always)]
    pub fn norm(&self) -> BtScalar { return self.length(); }

    /// Return the norm (length) of the vector
    #[inline(always)]
    pub fn safe_norm(&self) -> BtScalar {
        let d = self.length2();
        // workaround for some clang/gcc issue of sqrtf(tiny number) = -INF
        if d > SIMD_EPSILON {
            return d.sqrt();
        }
        return 0.0;
    }

    /// Return the distance squared between the ends of this and another vector
    /// This is symantically treating the vector like a point
    #[inline(always)]
    fn distance2(&self, v: BtVector3) -> BtScalar {
        return (v - self).length2();
    }

    /// Return the distance between the ends of this and another vector
    /// This is symantically treating the vector like a point
    #[inline(always)]
    fn distance(&self, v: BtVector3) -> BtScalar {
        return (v - self).length();
    }

    #[inline(always)]
    pub fn safe_normalize(&mut self) {
        let l2 = self.length2();

        if l2 >= SIMD_EPSILON * SIMD_EPSILON {
            self.div_assign(l2.sqrt());
        } else {
            self.set_value(1.0, 0.0, 0.0);
        }
    }

    /// Normalize this vector
    /// * x^2 + y^2 + z^2 = 1
    #[inline(always)]
    fn normalize(&mut self) -> &mut BtVector3 {
        unsafe {
            // dot product first
            let mut vd = _mm_mul_ps(self.m_vec128.simd, self.m_vec128.simd);
            let mut z = _mm_movehl_ps(vd, vd);
            let mut y = _mm_shuffle_ps(vd, vd, 0x55);
            vd = _mm_add_ss(vd, y);
            vd = _mm_add_ss(vd, z);

            // NR step 1/sqrt(x) - vd is x, y is output
            y = _mm_rsqrt_ss(vd); // estimate

            //  one step NR
            z = v_1_5!();
            vd = _mm_mul_ss(vd, v_half!()); // vd * 0.5
            // x2 = vd;
            vd = _mm_mul_ss(vd, y); // vd * 0.5 * y0
            vd = _mm_mul_ss(vd, y); // vd * 0.5 * y0 * y0
            z = _mm_sub_ss(z, vd);  // 1.5 - vd * 0.5 * y0 * y0

            y = _mm_mul_ss(y, z); // y0 * (1.5 - vd * 0.5 * y0 * y0)

            y = bt_splat_ps!(y, 0x80);
            self.m_vec128.simd = _mm_mul_ps(self.m_vec128.simd, y);

            return self;
        }
    }

    // Return a normalized version of this vector
    #[inline(always)]
    fn normalized(&self) -> &BtVector3 {
        let nrm = self.clone();

        return nrm.normalized();
    }

    /// Return a rotated version of this vector
    /// - Parameter: w_axis The axis to rotate about
    /// - Parameter: angle The angle to rotate by
    #[inline(always)]
    #[allow(overflowing_literals)]
    fn rotate(&self, w_axis: BtVector3, _angle: BtScalar) -> BtVector3 {
        unsafe {
            let mut o = _mm_mul_ps(w_axis.m_vec128.simd, self.m_vec128.simd);
            let ssin = _angle.sin();
            let c = w_axis.cross(self).m_vec128.simd;
            o = _mm_and_ps(o, btv_fff0f_mask!());
            let scos = _angle.cos();

            let mut vsin = _mm_load_ss(&ssin); //    (S 0 0 0)
            let mut vcos = _mm_load_ss(&scos); //    (S 0 0 0)

            let y = bt_pshufd_ps!(o, 0xC9); //    (y z x 0)
            let z = bt_pshufd_ps!(o, 0xD2); //    (z x y 0)
            o = _mm_add_ps(o, y);
            vsin = bt_pshufd_ps!(vsin, 0x80); //    (S S S 0)
            o = _mm_add_ps(o, z);
            vcos = bt_pshufd_ps!(vcos, 0x80); //    (S S S 0)

            vsin = _mm_mul_ps(vsin, c);
            o = _mm_mul_ps(o, w_axis.m_vec128.simd);
            let x = _mm_sub_ps(self.m_vec128.simd, o);

            o = _mm_add_ps(o, vsin);
            vcos = _mm_mul_ps(vcos, x);
            o = _mm_add_ps(o, vcos);

            return BtVector3::new_simd(o);
        }
    }

    // /// Return the angle between this and another vector
    // /// - Parameter: v The other vector
    // fn angle(v: BtVector3) -> BtScalar {
    //     let s = btSqrt(length2() * v.length2());
    //     assert!(s != 0.0);
    //     return btAcos(dot(v) / s);
    // }
    //
    // /// Return a vector with the absolute values of each element
    // fn absolute() -> BtVector3 {
    //     unsafe {
    //         return btVector3(_mm_and_ps(mVec128, btv3AbsfMask));
    //     }
    // }

    /// Return the cross product between this and another vector
    /// - Parameter: v The other vector */
    fn cross(&self, v: &BtVector3) -> BtVector3 {
        unsafe {
            let mut t = bt_pshufd_ps!(self.m_vec128.simd, bt_shuffle!(1, 2, 0, 3));   //    (Y Z X 0)
            let mut vv = bt_pshufd_ps!(v.m_vec128.simd, bt_shuffle!(1, 2, 0, 3)); //    (Y Z X 0)

            vv = _mm_mul_ps(vv, self.m_vec128.simd);
            t = _mm_mul_ps(t, v.m_vec128.simd);
            vv = _mm_sub_ps(vv, t);

            vv = bt_pshufd_ps!(vv,  bt_shuffle!(1, 2, 0, 3));
            return BtVector3::new_simd(vv);
        }
    }

    // fn triple(v1:BtVector3, v2:BtVector3) ->BtScalar {
    //
    // }

    #[inline(always)]
    pub fn set_value(&mut self, _x: BtScalar, _y: BtScalar, _z: BtScalar) {
        self.m_vec128.array = [_x, _y, _z, 0.0]
    }
}

#[test]
pub fn test() {
    let mut a = BtVector3::new_default();
    let b: BtScalar = 10.0;
    a /= b;

    a.set_value(10.0, 20.0, 4.0);
    unsafe {
        assert_eq!(a.m_vec128.array[0], 10.0);
    }
}

















