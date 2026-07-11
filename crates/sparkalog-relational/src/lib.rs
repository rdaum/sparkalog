//! Backend-neutral relational algebra over Sparkalog's canonical storage.

pub use sparkalog_storage::{Column, Relation2};

/// The semi-naive view of a logical relation consumed by an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationVersion {
    Full,
    Delta,
    Newt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColumnRef {
    pub relation: u32,
    pub column: u32,
    pub version: RelationVersion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JoinKey {
    pub left: ColumnRef,
    pub right: ColumnRef,
}

/// Operations for which the execution crate will eventually select a native
/// Rust or CUDA implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorKind {
    Scan,
    Filter,
    Project,
    Join,
    AntiJoin,
    Union,
    Distinct,
    Sort,
    Reduce,
    Persist,
}
