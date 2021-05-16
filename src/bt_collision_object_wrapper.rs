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
    m_parent: Box<BtCollisionObjectWrapper>,
    m_part_id: i32,
    m_index: i32,
}