//! Semi-naive fixpoint orchestration over relational plans.

pub use sparkalog_relational::RelationVersion;

use sparkalog_execution::{
    AntiJoinPlacementPolicy, CudaStream, DistinctPlacementPolicy, InputProvenance,
    JoinPlacementPolicy, Placement, UnionPlacementPolicy, anti_join_auto, distinct_auto, join_auto,
    union_auto,
};
use sparkalog_relational::{
    BinaryDistinct, BinaryEqualityJoin, JoinInput, JoinProjection, SortedBinaryAntiJoin,
    SortedBinaryUnion,
};
use sparkalog_storage::{
    AntiJoinWorkspace, DistinctWorkspace, JoinWorkspace, RelationBuffer, RelationView,
    U32RangeIndex, UnionWorkspace,
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
pub struct IterationPlacements {
    pub join: Placement,
    pub distinct: Placement,
    pub anti_join: Placement,
    pub union: Placement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IterationResult {
    pub candidate_rows: usize,
    pub distinct_candidate_rows: usize,
    pub newt_rows: usize,
    pub full_rows: usize,
    pub full_provenance: InputProvenance,
    pub placements: IterationPlacements,
}

/// Reusable intermediates and outputs for one semi-naive transitive-closure
/// step. Ping-pong two instances to feed one step's `NEWT` and `FULL` into the
/// next without copying canonical columns.
pub struct TransitiveClosureStep {
    candidates: JoinWorkspace,
    distinct_candidates: DistinctWorkspace,
    newt: AntiJoinWorkspace,
    full: UnionWorkspace,
}

impl TransitiveClosureStep {
    pub fn new() -> sparkalog_storage::Result<Self> {
        Ok(Self {
            candidates: JoinWorkspace::new(2)?,
            distinct_candidates: DistinctWorkspace::new()?,
            newt: AntiJoinWorkspace::new()?,
            full: UnionWorkspace::new()?,
        })
    }

    pub fn newt(&self) -> &RelationBuffer {
        self.newt.output()
    }

    pub fn full(&self) -> &RelationBuffer {
        self.full.output()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        delta: RelationView<'_>,
        edge: RelationView<'_>,
        full: RelationView<'_>,
        edge_index: &U32RangeIndex,
        full_provenance: InputProvenance,
        stream: Option<&CudaStream>,
        policies: IterationPolicies,
    ) -> sparkalog_execution::Result<IterationResult> {
        let join = join_auto(
            delta,
            edge,
            edge_index,
            join_plan(),
            &mut self.candidates,
            stream,
            policies.join,
        )?;
        let join_provenance = placement_provenance(join);
        let distinct = distinct_auto(
            self.candidates.output().view(),
            distinct_plan(),
            join_provenance,
            &mut self.distinct_candidates,
            stream,
            policies.distinct,
        )?;
        let distinct_provenance = placement_provenance(distinct);
        let anti_input_provenance = combine_provenance(distinct_provenance, full_provenance);
        let anti_join = anti_join_auto(
            self.distinct_candidates.output().view(),
            full,
            anti_join_plan(),
            anti_input_provenance,
            &mut self.newt,
            stream,
            policies.anti_join,
        )?;
        let newt_provenance = placement_provenance(anti_join);
        let union_input_provenance = combine_provenance(full_provenance, newt_provenance);
        let union = union_auto(
            full,
            self.newt.output().view(),
            union_plan(),
            union_input_provenance,
            &mut self.full,
            stream,
            policies.union,
        )?;
        Ok(IterationResult {
            candidate_rows: self.candidates.output().len(),
            distinct_candidate_rows: self.distinct_candidates.output().len(),
            newt_rows: self.newt.output().len(),
            full_rows: self.full.output().len(),
            full_provenance: placement_provenance(union),
            placements: IterationPlacements {
                join,
                distinct,
                anti_join,
                union,
            },
        })
    }
}

fn placement_provenance(placement: Placement) -> InputProvenance {
    match placement {
        Placement::Gpu => InputProvenance::Gpu,
        Placement::CpuSerial | Placement::CpuParallel => InputProvenance::Cpu,
    }
}

fn combine_provenance(left: InputProvenance, right: InputProvenance) -> InputProvenance {
    if left == InputProvenance::Gpu && right == InputProvenance::Gpu {
        InputProvenance::Gpu
    } else {
        InputProvenance::Cpu
    }
}

fn join_plan() -> BinaryEqualityJoin {
    BinaryEqualityJoin {
        left_key: 1,
        right_key: 0,
        output: [
            JoinProjection {
                input: JoinInput::Left,
                column: 0,
            },
            JoinProjection {
                input: JoinInput::Right,
                column: 1,
            },
        ],
    }
}

fn distinct_plan() -> BinaryDistinct {
    BinaryDistinct { columns: [0, 1] }
}

fn anti_join_plan() -> SortedBinaryAntiJoin {
    SortedBinaryAntiJoin {
        left: [0, 1],
        right: [0, 1],
    }
}

fn union_plan() -> SortedBinaryUnion {
    SortedBinaryUnion {
        left: [0, 1],
        right: [0, 1],
    }
}

/// Cardinality state at a semi-naive iteration boundary.
///
/// `next_delta_rows` is expected to have already been deduplicated both
/// internally and against `full_rows` by the relational plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixpointState {
    full_rows: usize,
    delta_rows: usize,
    iterations: usize,
}

impl FixpointState {
    pub fn seeded(seed_rows: usize) -> Self {
        Self {
            full_rows: seed_rows,
            delta_rows: seed_rows,
            iterations: 0,
        }
    }

    pub fn full_rows(self) -> usize {
        self.full_rows
    }

    pub fn delta_rows(self) -> usize {
        self.delta_rows
    }

    pub fn iterations(self) -> usize {
        self.iterations
    }

    pub fn reached_fixpoint(self) -> bool {
        self.delta_rows == 0
    }

    pub fn advance(&mut self, next_delta_rows: usize) {
        self.full_rows += next_delta_rows;
        self.delta_rows = next_delta_rows;
        self.iterations += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparkalog_storage::Relation;

    fn relation2(rows: &[(u32, u32)]) -> Relation {
        let mut relation = Relation::new(2, rows.len()).unwrap();
        for (row, &(first, second)) in rows.iter().enumerate() {
            relation.column_mut(0).unwrap().as_mut_slice()[row] = first;
            relation.column_mut(1).unwrap().as_mut_slice()[row] = second;
        }
        relation
    }

    #[test]
    fn empty_delta_ends_the_fixpoint() {
        let mut state = FixpointState::seeded(4);
        state.advance(2);
        state.advance(0);

        assert!(state.reached_fixpoint());
        assert_eq!(state.full_rows(), 6);
        assert_eq!(state.iterations(), 2);
    }

    #[test]
    fn complete_transitive_closure_step_produces_newt_and_updated_full() {
        let edge = relation2(&[(1, 2), (2, 3), (3, 4)]);
        let index = U32RangeIndex::build(edge.column(0).unwrap()).unwrap();
        let mut step = TransitiveClosureStep::new().unwrap();

        let result = step
            .execute(
                edge.view(),
                edge.view(),
                edge.view(),
                &index,
                InputProvenance::Cpu,
                None,
                IterationPolicies::default(),
            )
            .unwrap();

        assert_eq!(result.candidate_rows, 2);
        assert_eq!(result.distinct_candidate_rows, 2);
        assert_eq!(result.newt_rows, 2);
        assert_eq!(result.full_rows, 5);
        assert_eq!(step.newt().view().column_slice(0).unwrap(), &[1, 2]);
        assert_eq!(step.newt().view().column_slice(1).unwrap(), &[3, 4]);
        assert_eq!(
            step.full().view().column_slice(0).unwrap(),
            &[1, 1, 2, 2, 3]
        );
        assert_eq!(
            step.full().view().column_slice(1).unwrap(),
            &[2, 3, 3, 4, 4]
        );

        let mut state = FixpointState::seeded(edge.len());
        state.advance(result.newt_rows);
        assert_eq!(state.full_rows(), result.full_rows);
        assert_eq!(state.delta_rows(), result.newt_rows);
    }
}
