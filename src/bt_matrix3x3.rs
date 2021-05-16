/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

use crate::bt_vector3::BtVector3;

/// The btMatrix3x3 class implements a 3x3 rotation matrix, to perform linear algebra in combination with
/// btQuaternion, btTransform and btVector3. Make sure to only include a pure orthogonal matrix without scaling. */
pub struct BtMatrix3x3 {
    m_el: [BtVector3; 3],
}

impl BtMatrix3x3 {
    /// No initialization constructor
    pub fn new_default() -> BtMatrix3x3 {
        return BtMatrix3x3 {
            m_el: [BtVector3::new_default(), BtVector3::new_default(), BtVector3::new_default()]
        };
    }
}

impl BtMatrix3x3 {
    /// Get a column of the matrix as a vector
    /// - Parameter: i Column number 0 indexed */
    #[inline(always)]
    pub fn get_column(&self, i: usize) -> BtVector3 { return BtVector3::new(self.m_el[0][i], self.m_el[1][i], self.m_el[2][i]); }
}