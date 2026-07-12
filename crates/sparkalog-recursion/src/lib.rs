//! Semi-naive fixpoint orchestration over relational plans.

mod generic;

pub use generic::{
    GenericExecutionError, RecursiveExecutor, RelationStore, RelationStoreError, SccSummary,
    transitive_closure_scc,
};
pub use sparkalog_relational::RelationVersion;

use sparkalog_execution::{
    AntiJoinPlacementPolicy, DistinctPlacementPolicy, InputProvenance, JoinPlacementPolicy,
    Placement, UnionPlacementPolicy,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IterationPolicies {
    pub join: JoinPlacementPolicy,
    pub distinct: DistinctPlacementPolicy,
    pub anti_join: AntiJoinPlacementPolicy,
    pub union: UnionPlacementPolicy,
}

impl Default for IterationPolicies {
    fn default() -> Self {
        Self {
            join: JoinPlacementPolicy::MEASURED_GB10_DBLP,
            distinct: DistinctPlacementPolicy::MEASURED_GB10_DBLP,
            anti_join: AntiJoinPlacementPolicy::MEASURED_GB10_DBLP,
            union: UnionPlacementPolicy::MEASURED_GB10_DBLP,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClausePlacements {
    pub rule_index: usize,
    pub join: Placement,
    pub distinct: Placement,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetPlacements {
    pub clauses: Vec<ClausePlacements>,
    pub contribution_unions: Vec<Placement>,
    pub anti_join: Placement,
    pub union: Placement,
}

pub(crate) fn placement_provenance(placement: Placement) -> InputProvenance {
    match placement {
        Placement::Gpu => InputProvenance::Gpu,
        Placement::CpuSerial | Placement::CpuParallel => InputProvenance::Cpu,
    }
}

pub(crate) fn combine_provenance(left: InputProvenance, right: InputProvenance) -> InputProvenance {
    if left == InputProvenance::Gpu && right == InputProvenance::Gpu {
        InputProvenance::Gpu
    } else {
        InputProvenance::Cpu
    }
}
