use crate::bt_scalar::{*};
use std::arch::x86_64::{*};
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign, Add};

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

//--------------------------------------------------------------------------------------------------
pub struct BtVector3 {
    m_vec128: BtSimdFloat4,
}

impl BtVector3 {
    #[inline(always)]
    pub fn get128(&self) -> BtSimdFloat4 { return self.m_vec128; }

    #[inline(always)]
    pub fn set128(&mut self, v128: BtSimdFloat4) { self.m_vec128 = v128; }

    pub fn new_default() -> BtVector3 {
        return BtVector3 {
            m_vec128: unsafe { _mm_set_ps(0.0, 0.0, 0.0, 0.0) }
        };
    }

    /// Constructor from scalars
    pub fn new(_x: BtScalar, _y: BtScalar, _z: BtScalar) -> BtVector3 {
        return BtVector3 {
            m_vec128: unsafe { _mm_set_ps(_x, _y, _z, 0.0) }
        };
    }
}

impl AddAssign for BtVector3 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.m_vec128 = unsafe { _mm_sub_ps(self.m_vec128, rhs.m_vec128) }
    }
}

impl SubAssign for BtVector3 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.m_vec128 = unsafe { _mm_sub_ps(self.m_vec128, rhs.m_vec128) }
    }
}

impl MulAssign<BtScalar> for BtVector3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: BtScalar) {
        unsafe {
            let mut vs = _mm_load_ss(&rhs); //    (S 0 0 0)
            vs = bt_pshufd_ps!(vs, 0x80); //    (S S S 0.0)
            self.m_vec128 = _mm_mul_ps(self.m_vec128, vs)
        }
    }
}

impl MulAssign<&BtScalar> for BtVector3 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &BtScalar) {
        unsafe {
            let mut vs = _mm_load_ss(&*rhs); //    (S 0 0 0)
            vs = bt_pshufd_ps!(vs, 0x80); //    (S S S 0.0)
            self.m_vec128 = _mm_mul_ps(self.m_vec128, vs)
        }
    }
}

impl DivAssign<BtScalar> for BtVector3 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: BtScalar) {
        self.mul_assign(1.0 / rhs)
    }
}

impl BtVector3 {
    #[inline(always)]
    pub fn dot(&self, v: &BtVector3) -> BtScalar {
        unsafe {
            let mut vd = _mm_mul_ps(self.m_vec128, v.m_vec128);
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
        unsafe {
            return self.length2().sqrt();
        }
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

    // /// Return the distance squared between the ends of this and another vector
    // /// This is symantically treating the vector like a point
    // #[inline(always)]
    // fn distance2(&self, v: BtVector3) -> BtScalar {}
    //
    // /// Return the distance between the ends of this and another vector
    // /// This is symantically treating the vector like a point
    // #[inline(always)]
    // fn distance(&self, v: BtVector3) -> BtScalar {}

    #[inline(always)]
    pub fn safe_normalize(&mut self) -> &mut BtVector3 {
        let l2 = self.length2();

        if l2 >= SIMD_EPSILON * SIMD_EPSILON {
            self.div_assign(l2.sqrt());
        } else {
            self.set_value(1.0, 0.0, 0.0);
        }
        return self;
    }

    #[inline(always)]
    pub fn set_value(&mut self, _x: BtScalar, _y: BtScalar, _z: BtScalar) {
        unsafe { self.m_vec128 = _mm_set_ps(_x, _y, _z, 0.0) }
    }
}

#[test]
pub fn test() {
    let mut a = BtVector3::new_default();
    let b: BtScalar = 10.0;
    a /= b;

    a.set_value(10.0, 20.0, 4.0);
}

















