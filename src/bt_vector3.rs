use crate::bt_scalar::{*};
use crate::bt_casti_to128f;
use crate::bt_castf_to128d;
use crate::bt_castd_to128f;
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
macro_rules! btv_xyz_maskf {
    () => {
        btv_fff0f_mask!()
    };
}

#[macro_export]
macro_rules! btv_absf_mask {
    () => {
        bt_casti_to128f!(btv_abs_mask!())
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

macro_rules! impl_vector_ops {
    ($Vector:ty, $VecNew:path) => {
        impl AddAssign for $Vector {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                self.m_vec128.simd = unsafe { _mm_sub_ps(self.m_vec128.simd, rhs.m_vec128.simd) }
            }
        }

        impl SubAssign for $Vector {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                self.m_vec128.simd = unsafe { _mm_sub_ps(self.m_vec128.simd, rhs.m_vec128.simd) }
            }
        }

        impl MulAssign<BtScalar> for $Vector {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: BtScalar) {
                unsafe {
                    let mut vs = _mm_load_ss(&rhs); //    (S 0 0 0)
                    vs = bt_pshufd_ps!(vs, 0x80); //    (S S S 0.0)
                    self.m_vec128.simd = _mm_mul_ps(self.m_vec128.simd, vs)
                }
            }
        }

        impl MulAssign<&BtScalar> for $Vector {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: &BtScalar) {
                unsafe {
                    let mut vs = _mm_load_ss(&*rhs); //    (S 0 0 0)
                    vs = bt_pshufd_ps!(vs, 0x80); //    (S S S 0.0)
                    self.m_vec128.simd = _mm_mul_ps(self.m_vec128.simd, vs)
                }
            }
        }

        impl MulAssign for $Vector {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: $Vector) {
                unsafe {
                    self.m_vec128.simd = _mm_mul_ps(self.m_vec128.simd, rhs.m_vec128.simd);
                }
            }
        }

        impl DivAssign<BtScalar> for $Vector {
            #[inline(always)]
            fn div_assign(&mut self, rhs: BtScalar) {
                self.mul_assign(1.0 / rhs)
            }
        }

        macro_rules! impl_bin_add_vector {
            ($lhs:ty, $rhs:ty) => {
                /// Return the sum of two vectors (Point symantics)
                impl Add<$rhs> for $lhs {
                    type Output = $Vector;

                    fn add(self, rhs: $rhs) -> Self::Output {
                        unsafe {
                            return $VecNew(_mm_add_ps(self.m_vec128.simd, rhs.m_vec128.simd));
                        }
                    }
                }
            }
        }

        impl_bin_add_vector!($Vector, $Vector);
        impl_bin_add_vector!(&$Vector, BtVector3);
        impl_bin_add_vector!($Vector, &$Vector);
        impl_bin_add_vector!(&$Vector, &$Vector);

        macro_rules! impl_bin_mul_vector {
            ($lhs:ty, $rhs:ty) => {
                /// Return the sum of two vectors (Point symantics)
                impl Mul<$rhs> for $lhs {
                    type Output = $Vector;

                    fn mul(self, rhs: $rhs) -> Self::Output {
                        unsafe {
                            return $VecNew(_mm_mul_ps(self.m_vec128.simd, rhs.m_vec128.simd));
                        }
                    }
                }
            }
        }

        impl_bin_mul_vector!($Vector, $Vector);
        impl_bin_mul_vector!(&$Vector, $Vector);
        impl_bin_mul_vector!($Vector, &$Vector);
        impl_bin_mul_vector!(&$Vector, &$Vector);

        macro_rules! impl_bin_sub_vector {
            ($lhs:ty, $rhs:ty) => {
                // Return the difference between two vectors
                impl Sub<$rhs> for $lhs {
                    type Output = $Vector;

                    #[allow(overflowing_literals)]
                    fn sub(self, rhs: $rhs) -> Self::Output {
                        unsafe {
                            let r = _mm_sub_ps(self.m_vec128.simd, rhs.m_vec128.simd);
                            let b = btv_fff0f_mask!();
                            return $VecNew(_mm_and_ps(r, b));
                        }
                    }
                }
            }
        }

        impl_bin_sub_vector!($Vector, $Vector);
        impl_bin_sub_vector!(&$Vector, $Vector);
        impl_bin_sub_vector!($Vector, &$Vector);
        impl_bin_sub_vector!(&$Vector, &$Vector);

        /// Return the negative of the vector
        impl Neg for $Vector {
            type Output = $Vector;

            #[allow(overflowing_literals)]
            fn neg(self) -> Self::Output {
                unsafe {
                    let r = _mm_xor_ps(self.m_vec128.simd, btv_mzero_mask!());
                    return $VecNew(_mm_and_ps(r, btv_fff0f_mask!()));
                }
            }
        }

        impl Mul<BtScalar> for $Vector {
            type Output = $Vector;

            fn mul(self, rhs: f32) -> Self::Output {
                unsafe {
                    let mut vs = _mm_load_ss(&rhs); //    (S 0 0 0)
                    vs = bt_pshufd_ps!(vs, 0x80); //    (S S S 0.0)
                    return $VecNew(_mm_mul_ps(self.m_vec128.simd, vs));
                }
            }
        }

        impl Div<BtScalar> for $Vector {
            type Output = $Vector;

            fn div(self, rhs: f32) -> Self::Output {
                return self * 1.0 / rhs;
            }
        }

        impl Div for $Vector {
            type Output = $Vector;

            #[allow(overflowing_literals)]
            fn div(self, rhs: Self) -> Self::Output {
                unsafe {
                    let mut vec = _mm_div_ps(self.m_vec128.simd, rhs.m_vec128.simd);
                    vec = _mm_and_ps(vec, btv_fff0f_mask!());
                    return $VecNew(vec);
                }
            }
        }

        impl PartialEq for $Vector {
            fn eq(&self, other: &Self) -> bool {
                unsafe {
                    return 0xf == _mm_movemask_ps(_mm_cmpeq_ps(self.m_vec128.simd, other.m_vec128.simd));
                }
            }
        }

        impl Eq for $Vector {}
    }
}

impl_vector_ops!(BtVector3, BtVector3::new_simd);

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
    pub fn distance2(&self, v: &BtVector3) -> BtScalar {
        return (v - self).length2();
    }

    /// Return the distance between the ends of this and another vector
    /// This is symantically treating the vector like a point
    #[inline(always)]
    pub fn distance(&self, v: &BtVector3) -> BtScalar {
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
    pub fn normalize(&mut self) -> &mut BtVector3 {
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
    pub fn normalized(&self) -> &BtVector3 {
        let nrm = self.clone();

        return nrm.normalized();
    }

    /// Return a rotated version of this vector
    /// - Parameter: w_axis The axis to rotate about
    /// - Parameter: angle The angle to rotate by
    #[inline(always)]
    #[allow(overflowing_literals)]
    pub fn rotate(&self, w_axis: BtVector3, _angle: BtScalar) -> BtVector3 {
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

    /// Return the angle between this and another vector
    /// - Parameter: v The other vector
    #[inline(always)]
    pub fn angle(&self, v: &BtVector3) -> BtScalar {
        let s = (self.length2() * v.length2()).sqrt();
        return (self.dot(v) / s).cos();
    }

    /// Return a vector with the absolute values of each element
    #[inline(always)]
    pub fn absolute(&self) -> BtVector3 {
        unsafe {
            return BtVector3::new_simd(_mm_and_ps(self.m_vec128.simd, btv3_absf_mask!()));
        }
    }

    /// Return the cross product between this and another vector
    /// - Parameter: v The other vector */
    #[inline(always)]
    pub fn cross(&self, v: &BtVector3) -> BtVector3 {
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

    #[inline(always)]
    pub fn triple(&self, v1: &BtVector3, v2: &BtVector3) -> BtScalar {
        unsafe {
            // cross:
            let mut t = _mm_shuffle_ps(v1.m_vec128.simd, v1.m_vec128.simd, bt_shuffle!(1, 2, 0, 3)); //    (Y Z X 0)
            let mut v = _mm_shuffle_ps(v2.m_vec128.simd, v2.m_vec128.simd, bt_shuffle!(1, 2, 0, 3)); //    (Y Z X 0)

            v = _mm_mul_ps(v, v1.m_vec128.simd);
            t = _mm_mul_ps(t, v2.m_vec128.simd);
            v = _mm_sub_ps(v, t);

            v = _mm_shuffle_ps(v, v, bt_shuffle!(1, 2, 0, 3));

            // dot:
            v = _mm_mul_ps(v, self.m_vec128.simd);
            let z = _mm_movehl_ps(v, v);
            let y = _mm_shuffle_ps(v, v, 0x55);
            v = _mm_add_ss(v, y);
            v = _mm_add_ss(v, z);
            return _mm_cvtss_f32(v);
        }
    }

    /// Return the axis with the smallest value
    /// Note return values are 0,1,2 for x, y, or z
    #[inline(always)]
    pub fn min_axis(&self) -> i32 {
        unsafe {
            return match self.m_vec128.array[0] < self.m_vec128.array[1] {
                true => {
                    match self.m_vec128.array[0] < self.m_vec128.array[2] {
                        true => { 0 }
                        false => { 2 }
                    }
                }
                false => {
                    match self.m_vec128.array[1] < self.m_vec128.array[2] {
                        true => { 1 }
                        false => { 2 }
                    }
                }
            };
        }
    }

    /// Return the axis with the largest value
    /// Note return values are 0,1,2 for x, y, or z
    #[inline(always)]
    pub fn max_axis(&self) -> i32 {
        unsafe {
            return match self.m_vec128.array[0] < self.m_vec128.array[1] {
                true => {
                    match self.m_vec128.array[1] < self.m_vec128.array[2] {
                        true => { 2 }
                        false => { 1 }
                    }
                }
                false => {
                    match self.m_vec128.array[0] < self.m_vec128.array[2] {
                        true => { 2 }
                        false => { 0 }
                    }
                }
            };
        }
    }

    #[inline(always)]
    pub fn furthest_axis(&self) -> i32 { return self.absolute().min_axis(); }

    #[inline(always)]
    pub fn closest_axis(&self) -> i32 { return self.absolute().max_axis(); }

    #[inline(always)]
    pub fn set_interpolate3(&mut self, v0: BtVector3, v1: BtVector3, rt: BtScalar) {
        unsafe {
            let mut vrt = _mm_load_ss(&rt); //    (rt 0 0 0)
            let s = 1.0 - rt;
            let mut vs = _mm_load_ss(&s); //    (S 0 0 0)
            vs = bt_pshufd_ps!(vs, 0x80); //    (S S S 0.0)
            let r0 = _mm_mul_ps(v0.m_vec128.simd, vs);
            vrt = bt_pshufd_ps!(vrt, 0x80); //    (rt rt rt 0.0)
            let r1 = _mm_mul_ps(v1.m_vec128.simd, vrt);
            let tmp3 = _mm_add_ps(r0, r1);
            self.m_vec128.simd = tmp3;
        }
    }

    /// Return the linear interpolation between this and another vector
    /// - Parameter:  v The other vector
    /// - Parameter:  t The ration of this to v (t = 0 => return this, t=1 => return other)
    #[inline(always)]
    pub fn lerp(&self, v: &BtVector3, t: &BtScalar) -> BtVector3 {
        unsafe {
            let mut vt = _mm_load_ss(t); //    (t 0 0 0)
            vt = bt_pshufd_ps!(vt, 0x80); //    (rt rt rt 0.0)
            let mut vl = _mm_sub_ps(v.m_vec128.simd, self.m_vec128.simd);
            vl = _mm_mul_ps(vl, vt);
            vl = _mm_add_ps(vl, self.m_vec128.simd);

            return BtVector3::new_simd(vl);
        }
    }

    /// Return the x value */
    #[inline(always)]
    pub fn get_x(&self) -> BtScalar { unsafe { return self.m_vec128.array[0]; } }

    /// Return the y value */
    #[inline(always)]
    pub fn get_y(&self) -> BtScalar { unsafe { return self.m_vec128.array[1]; } }

    /// Return the z value */
    #[inline(always)]
    pub fn get_z(&self) -> BtScalar { unsafe { return self.m_vec128.array[2]; } }

    /// Set the x value */
    #[inline(always)]
    pub fn set_x(&mut self, _x: BtScalar) { unsafe { self.m_vec128.array[0] = _x; } }

    /// Set the y value */
    #[inline(always)]
    pub fn set_y(&mut self, _y: BtScalar) { unsafe { self.m_vec128.array[1] = _y; } }

    /// Set the z value */
    #[inline(always)]
    pub fn set_z(&mut self, _z: BtScalar) { unsafe { self.m_vec128.array[2] = _z; } }

    /// Set the w value */
    #[inline(always)]
    pub fn set_w(&mut self, _w: BtScalar) { unsafe { self.m_vec128.array[3] = _w; } }

    /// Return the x value */
    #[inline(always)]
    pub fn x(&self) -> BtScalar { unsafe { return self.m_vec128.array[0]; } }

    /// Return the y value */
    #[inline(always)]
    pub fn y(&self) -> BtScalar { unsafe { return self.m_vec128.array[1]; } }

    /// Return the z value */
    #[inline(always)]
    pub fn z(&self) -> BtScalar { unsafe { return self.m_vec128.array[2]; } }

    /// Return the w value */
    #[inline(always)]
    pub fn w(&self) -> BtScalar { unsafe { return self.m_vec128.array[3]; } }

    /// Set each element to the max of the current values and the values of another btVector3
    /// - Parameter:  other The other btVector3 to compare with
    #[inline(always)]
    pub fn set_max(&mut self, other: BtVector3) {
        unsafe {
            self.m_vec128.simd = _mm_max_ps(self.m_vec128.simd, other.m_vec128.simd);
        }
    }

    /// Set each element to the min of the current values and the values of another btVector3
    /// - Parameter:  other The other btVector3 to compare with
    #[inline(always)]
    pub fn set_min(&mut self, other: BtVector3) {
        unsafe {
            self.m_vec128.simd = _mm_min_ps(self.m_vec128.simd, other.m_vec128.simd);
        }
    }

    /// Set the values
    /// - Parameter: x Value of x
    /// - Parameter: y Value of y
    /// - Parameter: z Value of z
    #[inline(always)]
    pub fn set_value(&mut self, _x: BtScalar, _y: BtScalar, _z: BtScalar) {
        self.m_vec128.array = [_x, _y, _z, 0.0]
    }

    #[allow(overflowing_literals)]
    pub fn get_skew_symmetric_matrix(&self, v0: &mut BtVector3, v1: &mut BtVector3, v2: &mut BtVector3) {
        unsafe {
            let v = _mm_and_ps(self.m_vec128.simd, btv_fff0f_mask!());
            let mut vv0 = _mm_xor_ps(btv_mzero_mask!(), v);
            let mut vv2 = _mm_movelh_ps(vv0, v);

            let vv1 = _mm_shuffle_ps(v, vv0, 0xCE);

            vv0 = _mm_shuffle_ps(vv0, v, 0xDB);
            vv2 = _mm_shuffle_ps(vv2, v, 0xF9);

            v0.m_vec128.simd = vv0;
            v1.m_vec128.simd = vv1;
            v2.m_vec128.simd = vv2;
        }
    }

    pub fn set_zero(&mut self) {
        unsafe {
            self.m_vec128.simd = _mm_xor_ps(self.m_vec128.simd, self.m_vec128.simd);
        }
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        unsafe {
            return self.m_vec128.array[0] == 0.0 && self.m_vec128.array[1] == 0.0 && self.m_vec128.array[2] == 0.0;
        }
    }

    #[inline(always)]
    pub fn fuzzy_zero(&self) -> bool { return self.length2() < SIMD_EPSILON * SIMD_EPSILON; }

    /// returns index of maximum dot product between this and vectors in array[]
    /// - Parameter:  array The other vectors
    /// - Parameter:  array_count The number of other vectors
    /// - Parameter:  dotOut The maximum dot product */
    /// TODO: It can be accelerated by using SIMD.
    #[inline(always)]
    pub fn max_dot(&self, array: Vec<BtVector3>, array_count: i64, dot_out: &mut BtScalar) -> i64 {
        let mut max_dot1 = -SIMD_INFINITY;
        let mut pt_index = -1;
        for i in 0..array_count {
            let dot = array[i as usize].dot(self);

            if dot > max_dot1 {
                max_dot1 = dot;
                pt_index = i;
            }
        }

        *dot_out = max_dot1;
        return pt_index;
    }

    /// returns index of minimum dot product between this and vectors in array[]
    /// - Parameter:  array The other vectors
    /// - Parameter:  array_count The number of other vectors
    /// - Parameter:  dotOut The minimum dot product */
    /// TODO: It can be accelerated by using SIMD.
    #[inline(always)]
    pub fn min_dot(&self, array: Vec<BtVector3>, array_count: i64, dot_out: &mut BtScalar) -> i64 {
        let mut min_dot = SIMD_INFINITY;
        let mut pt_index = -1;

        for i in 0..array_count {
            let dot = array[i as usize].dot(self);

            if dot < min_dot {
                min_dot = dot;
                pt_index = i;
            }
        }

        *dot_out = min_dot;

        return pt_index;
    }

    /// create a vector as  btVector3( this->dot( btVector3 v0 ), this->dot( btVector3 v1), this->dot( btVector3 v2 ))
    /// # Examples
    ///
    /// ```
    /// use bullet_rs::bt_vector3::BtVector3;
    /// let a = BtVector3::new(1.0, 2.0, 3.0);
    /// let v0 = BtVector3::new(10.0, 0.0, 0.0);
    /// let v1 = BtVector3::new(0.0, 10.0, 0.0);
    /// let v2 = BtVector3::new(0.0, 0.0, 10.0);
    /// let result = a.dot3(v0, v1, v2);
    /// assert_eq!(result.x(), 10.0);
    /// assert_eq!(result.y(), 20.0);
    /// assert_eq!(result.z(), 30.0);
    /// ```
    ///
    #[inline(always)]
    #[allow(overflowing_literals)]
    pub fn dot3(&self, v0: BtVector3, v1: BtVector3, v2: BtVector3) -> BtVector3 {
        unsafe {
            let a0 = _mm_mul_ps(v0.m_vec128.simd, self.m_vec128.simd);
            let a1 = _mm_mul_ps(v1.m_vec128.simd, self.m_vec128.simd);
            let mut a2 = _mm_mul_ps(v2.m_vec128.simd, self.m_vec128.simd);
            let b0 = _mm_unpacklo_ps(a0, a1);
            let b1 = _mm_unpackhi_ps(a0, a1);
            let b2 = _mm_unpacklo_ps(a2, _mm_setzero_ps());
            let mut r = _mm_movelh_ps(b0, b2);
            r = _mm_add_ps(r, _mm_movehl_ps(b2, b0));
            a2 = _mm_and_ps(a2, btv_xyz_maskf!());
            r = _mm_add_ps(r, bt_castd_to128f!(_mm_move_sd( bt_castf_to128d!(a2), bt_castf_to128d!(b1))));
            return BtVector3::new_simd(r);
        }
    }
}

/// Return the dot product between two vectors
#[inline(always)]
pub fn bt_dot(v1: &BtVector3, v2: &BtVector3) -> BtScalar { return v1.dot(v2); }

/// Return the distance squared between two vectors */
#[inline(always)]
pub fn bt_distance2(v1: &BtVector3, v2: &BtVector3) -> BtScalar { return v1.distance2(v2); }

/// Return the distance between two vectors */
#[inline(always)]
pub fn bt_distance(v1: &BtVector3, v2: &BtVector3) -> BtScalar { return v1.distance(v2); }

/// Return the angle between two vectors */
#[inline(always)]
pub fn bt_angle(v1: &BtVector3, v2: &BtVector3) -> BtScalar { return v1.angle(v2); }

/// Return the cross product of two vectors */
#[inline(always)]
pub fn bt_cross(v1: &BtVector3, v2: &BtVector3) -> BtVector3 { return v1.cross(v2); }

#[inline(always)]
pub fn bt_triple(v1: &BtVector3, v2: &BtVector3, v3: &BtVector3) -> BtScalar {
    return v1.triple(v2, v3);
}

/// Return the linear interpolation between two vectors
/// - Parameter:  v1 One vector
/// - Parameter:  v2 The other vector
/// - Parameter:  t The ration of this to v (t = 0 => return v1, t=1 => return v2)
#[inline(always)]
pub fn lerp(v1: &BtVector3, v2: &BtVector3, t: &BtScalar) -> BtVector3 { return v1.lerp(v2, t); }

//--------------------------------------------------------------------------------------------------
pub struct BtVector4 {
    m_vec128: SimdToArray,
}

impl BtVector4 {
    #[inline(always)]
    pub fn get128(&self) -> BtSimdFloat4 { unsafe { return self.m_vec128.simd; } }

    #[inline(always)]
    pub fn set128(&mut self, v128: BtSimdFloat4) { self.m_vec128.simd = v128; }

    pub fn new_default() -> BtVector4 {
        return BtVector4 {
            m_vec128: SimdToArray { array: [0.0, 0.0, 0.0, 0.0] }
        };
    }

    /// Constructor from scalars
    pub fn new(_x: BtScalar, _y: BtScalar, _z: BtScalar, _w: BtScalar) -> BtVector4 {
        return BtVector4 {
            m_vec128: SimdToArray { array: [_x, _y, _z, _w] }
        };
    }

    /// Set Vector
    pub fn new_simd(v: BtSimdFloat4) -> BtVector4 {
        return BtVector4 {
            m_vec128: SimdToArray { simd: v }
        };
    }
}

impl_vector_ops!(BtVector4, BtVector4::new_simd);

impl BtVector4 {
    /// Return a vector with the absolute values of each element
    #[inline(always)]
    pub fn absolute(&self) -> BtVector4 {
        unsafe {
            return BtVector4::new_simd(_mm_and_ps(self.m_vec128.simd, btv_absf_mask!()));
        }
    }

    /// Return the x value */
    #[inline(always)]
    pub fn get_x(&self) -> BtScalar { unsafe { return self.m_vec128.array[0]; } }

    /// Return the y value */
    #[inline(always)]
    pub fn get_y(&self) -> BtScalar { unsafe { return self.m_vec128.array[1]; } }

    /// Return the z value */
    #[inline(always)]
    pub fn get_z(&self) -> BtScalar { unsafe { return self.m_vec128.array[2]; } }

    #[inline(always)]
    pub fn get_w(&self) -> BtScalar { unsafe { return self.m_vec128.array[3]; } }

    /// Set the x value */
    #[inline(always)]
    pub fn set_x(&mut self, _x: BtScalar) { unsafe { self.m_vec128.array[0] = _x; } }

    /// Set the y value */
    #[inline(always)]
    pub fn set_y(&mut self, _y: BtScalar) { unsafe { self.m_vec128.array[1] = _y; } }

    /// Set the z value */
    #[inline(always)]
    pub fn set_z(&mut self, _z: BtScalar) { unsafe { self.m_vec128.array[2] = _z; } }

    /// Set the w value */
    #[inline(always)]
    pub fn set_w(&mut self, _w: BtScalar) { unsafe { self.m_vec128.array[3] = _w; } }

    /// Return the x value */
    #[inline(always)]
    pub fn x(&self) -> BtScalar { unsafe { return self.m_vec128.array[0]; } }

    /// Return the y value */
    #[inline(always)]
    pub fn y(&self) -> BtScalar { unsafe { return self.m_vec128.array[1]; } }

    /// Return the z value */
    #[inline(always)]
    pub fn z(&self) -> BtScalar { unsafe { return self.m_vec128.array[2]; } }

    /// Return the w value */
    #[inline(always)]
    pub fn w(&self) -> BtScalar { unsafe { return self.m_vec128.array[3]; } }

    #[inline(always)]
    pub fn max_axis(&self) -> i32 {
        let mut max_index = -1;
        let mut max_val = -BT_LARGE_FLOAT;
        unsafe {
            if self.m_vec128.array[0] > max_val {
                max_index = 0;
                max_val = self.m_vec128.array[0];
            }
            if self.m_vec128.array[1] > max_val {
                max_index = 1;
                max_val = self.m_vec128.array[1];
            }
            if self.m_vec128.array[2] > max_val {
                max_index = 2;
                max_val = self.m_vec128.array[2];
            }
            if self.m_vec128.array[3] > max_val {
                max_index = 3;
            }
        }
        return max_index;
    }

    #[inline(always)]
    pub fn min_axis(&self) -> i32 {
        let mut min_index = -1;
        let mut min_val = BT_LARGE_FLOAT;
        unsafe {
            if self.m_vec128.array[0] < min_val {
                min_index = 0;
                min_val = self.m_vec128.array[0];
            }
            if self.m_vec128.array[1] < min_val {
                min_index = 1;
                min_val = self.m_vec128.array[1];
            }
            if self.m_vec128.array[2] < min_val {
                min_index = 2;
                min_val = self.m_vec128.array[2];
            }
            if self.m_vec128.array[3] < min_val {
                min_index = 3;
            }
        }
        return min_index;
    }

    #[inline(always)]
    pub fn closest_axis(&self) -> i32 { return self.absolute().max_axis(); }

    /// Set the values
    /// - Parameter: x Value of x
    /// - Parameter: y Value of y
    /// - Parameter: z Value of z
    /// - Parameter: w Value of w
    #[inline(always)]
    pub fn set_value(&mut self, _x: BtScalar, _y: BtScalar, _z: BtScalar, _w: BtScalar) {
        self.m_vec128.array = [_x, _y, _z, _w]
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














