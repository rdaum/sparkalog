//! Backend-neutral relational algebra over Sparkalog's canonical storage.

pub use sparkalog_storage::{
    AntiJoinWorkspace, Column, DistinctWorkspace, JoinWorkspace, Relation, RelationBuffer,
    RelationView, Selection, U32RangeIndex, UnionWorkspace,
};

/// The semi-naive view of a logical relation consumed by an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationVersion {
    Full,
    Delta,
    Newt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RelationId(pub u32);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinInput {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JoinProjection {
    pub input: JoinInput,
    pub column: u32,
}

/// A single-key equality join with a binary projected output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BinaryEqualityJoin {
    pub left_key: u32,
    pub right_key: u32,
    pub output: [JoinProjection; 2],
}

/// Sort and deduplicate a projected pair of `u32` columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BinaryDistinct {
    pub columns: [u32; 2],
}

/// Set difference between two lexicographically sorted binary relations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SortedBinaryAntiJoin {
    pub left: [u32; 2],
    pub right: [u32; 2],
}

/// Set union of two lexicographically sorted binary relations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SortedBinaryUnion {
    pub left: [u32; 2],
    pub right: [u32; 2],
}

/// One semi-naive binary rule lowered to relational operators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecursiveRulePlan {
    pub target: RelationId,
    pub delta_input: RelationId,
    pub right_input: RelationId,
    pub join: BinaryEqualityJoin,
    pub distinct: BinaryDistinct,
    pub anti_join: SortedBinaryAntiJoin,
    pub union: SortedBinaryUnion,
}

/// A strongly connected set of recursive relations evaluated round by round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecursiveSccPlan {
    pub relations: Vec<RelationId>,
    pub rules: Vec<RecursiveRulePlan>,
}

/// A rule-local binding carried through a general relational clause plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BindingId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanTerm {
    Binding(BindingId),
    Value(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedAtom {
    pub relation: RelationId,
    pub version: RelationVersion,
    pub terms: Vec<PlanTerm>,
}

/// A backend-neutral conjunction and projection. Positive atoms form a join
/// chain; negative atoms are evaluated as anti-joins after bindings exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalClausePlan {
    pub target: RelationId,
    pub head: Vec<PlanTerm>,
    pub positive: Vec<PlannedAtom>,
    pub negative: Vec<PlannedAtom>,
    /// Bindings retained after each positive atom, computed after join ordering.
    pub live_after: Vec<Vec<BindingId>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneralSccPlan {
    pub relations: Vec<RelationId>,
    pub recursive: bool,
    pub seeds: Vec<RelationalClausePlan>,
    pub recursive_variants: Vec<RelationalClausePlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneralStratumPlan {
    pub index: usize,
    pub sccs: Vec<GeneralSccPlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneralProgramPlan {
    pub predicate_count: usize,
    pub facts: Vec<(RelationId, Vec<u32>)>,
    pub strata: Vec<GeneralStratumPlan>,
    pub outputs: Vec<RelationId>,
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

    #[test]
    fn general_clause_plan_distinguishes_full_and_delta_inputs() {
        let plan = RelationalClausePlan {
            target: RelationId(0),
            head: vec![PlanTerm::Binding(BindingId(0))],
            positive: vec![PlannedAtom {
                relation: RelationId(0),
                version: RelationVersion::Delta,
                terms: vec![PlanTerm::Binding(BindingId(0))],
            }],
            negative: vec![],
            live_after: vec![vec![BindingId(0)]],
        };

        assert_eq!(plan.positive[0].version, RelationVersion::Delta);
    }
}
