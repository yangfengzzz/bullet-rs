/*
 * // Copyright (c) 2021 Feng Yang
 * //
 * // I am making my contributions/submissions to this project solely in my
 * // personal capacity and am not conveying any rights to any intellectual
 * // property of any third parties.
 */

/// btCollisionObjectWrappers an internal data structure.
/// Most users can ignore this and use btCollisionObject and btCollisionShape instead
pub struct BtCollisionObjectWrapper {
    pub m_parent: Box<BtCollisionObjectWrapper>,
    pub m_part_id: i32,
    pub m_index: i32,
}