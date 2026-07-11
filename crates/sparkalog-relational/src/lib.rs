//! Backend-neutral relational algebra over Sparkalog's canonical storage.

pub use sparkalog_storage::{Column, Relation, RelationView, Selection};

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

/// A comparison over one canonical `u32` column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum U32Predicate {
    Eq(u32),
    Ne(u32),
    Lt(u32),
    Le(u32),
    Gt(u32),
    Ge(u32),
}

impl U32Predicate {
    pub fn matches(self, value: u32) -> bool {
        match self {
            Self::Eq(expected) => value == expected,
            Self::Ne(expected) => value != expected,
            Self::Lt(upper) => value < upper,
            Self::Le(upper) => value <= upper,
            Self::Gt(lower) => value > lower,
            Self::Ge(lower) => value >= lower,
        }
    }
}

/// A backend-neutral filter which produces compact row identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Filter {
    pub column: ColumnRef,
    pub predicate: U32Predicate,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_u32_predicates_have_expected_boundary_semantics() {
        assert!(U32Predicate::Eq(4).matches(4));
        assert!(U32Predicate::Ne(4).matches(3));
        assert!(U32Predicate::Lt(4).matches(3));
        assert!(!U32Predicate::Lt(4).matches(4));
        assert!(U32Predicate::Le(4).matches(4));
        assert!(U32Predicate::Gt(4).matches(5));
        assert!(!U32Predicate::Gt(4).matches(4));
        assert!(U32Predicate::Ge(4).matches(4));
    }
}
