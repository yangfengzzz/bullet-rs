/// btCollisionObjectWrappers an internal data structure.
/// Most users can ignore this and use btCollisionObject and btCollisionShape instead
pub struct BtCollisionObjectWrapper {
    m_parent: Box<BtCollisionObjectWrapper>,
    m_part_id: i32,
    m_index: i32,
}