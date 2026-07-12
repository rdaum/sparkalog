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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepSlot {
    First,
    Second,
}

#[derive(Debug)]
pub enum FixpointError {
    Execution(sparkalog_execution::Error),
    IterationLimit {
        limit: usize,
        remaining_delta_rows: usize,
    },
}

impl std::fmt::Display for FixpointError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Execution(error) => error.fmt(formatter),
            Self::IterationLimit {
                limit,
                remaining_delta_rows,
            } => write!(
                formatter,
                "fixpoint did not converge within {limit} iterations; {remaining_delta_rows} delta rows remain"
            ),
        }
    }
}

impl std::error::Error for FixpointError {}

impl From<sparkalog_execution::Error> for FixpointError {
    fn from(error: sparkalog_execution::Error) -> Self {
        Self::Execution(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixpointSummary {
    pub state: FixpointState,
    pub full_provenance: InputProvenance,
    pub last_iteration: IterationResult,
}

/// CPU-scheduled semi-naive fixpoint driver. Two step workspaces alternate so
/// each iteration can borrow the prior iteration's canonical outputs directly.
pub struct FixpointDriver {
    first: TransitiveClosureStep,
    second: TransitiveClosureStep,
    final_slot: Option<StepSlot>,
}

impl FixpointDriver {
    pub fn new() -> sparkalog_storage::Result<Self> {
        Ok(Self {
            first: TransitiveClosureStep::new()?,
            second: TransitiveClosureStep::new()?,
            final_slot: None,
        })
    }

    pub fn full(&self) -> Option<&RelationBuffer> {
        match self.final_slot {
            Some(StepSlot::First) => Some(self.first.full()),
            Some(StepSlot::Second) => Some(self.second.full()),
            None => None,
        }
    }

    pub fn newt(&self) -> Option<&RelationBuffer> {
        match self.final_slot {
            Some(StepSlot::First) => Some(self.first.newt()),
            Some(StepSlot::Second) => Some(self.second.newt()),
            None => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        seed_delta: RelationView<'_>,
        edge: RelationView<'_>,
        seed_full: RelationView<'_>,
        edge_index: &U32RangeIndex,
        seed_full_provenance: InputProvenance,
        stream: Option<&CudaStream>,
        policies: IterationPolicies,
        max_iterations: usize,
    ) -> Result<FixpointSummary, FixpointError> {
        self.final_slot = None;
        if max_iterations == 0 {
            return Err(FixpointError::IterationLimit {
                limit: 0,
                remaining_delta_rows: seed_delta.len(),
            });
        }

        let mut last_iteration = self.first.execute(
            seed_delta,
            edge,
            seed_full,
            edge_index,
            seed_full_provenance,
            stream,
            policies,
        )?;
        let mut state = FixpointState::with_seed(seed_full.len(), seed_delta.len());
        state.advance(last_iteration.newt_rows);
        debug_assert_eq!(state.full_rows(), last_iteration.full_rows);
        let mut slot = StepSlot::First;

        while !state.reached_fixpoint() {
            if state.iterations() >= max_iterations {
                return Err(FixpointError::IterationLimit {
                    limit: max_iterations,
                    remaining_delta_rows: state.delta_rows(),
                });
            }
            last_iteration = match slot {
                StepSlot::First => self.second.execute(
                    self.first.newt().view(),
                    edge,
                    self.first.full().view(),
                    edge_index,
                    last_iteration.full_provenance,
                    stream,
                    policies,
                )?,
                StepSlot::Second => self.first.execute(
                    self.second.newt().view(),
                    edge,
                    self.second.full().view(),
                    edge_index,
                    last_iteration.full_provenance,
                    stream,
                    policies,
                )?,
            };
            slot = match slot {
                StepSlot::First => StepSlot::Second,
                StepSlot::Second => StepSlot::First,
            };
            state.advance(last_iteration.newt_rows);
            debug_assert_eq!(state.full_rows(), last_iteration.full_rows);
        }

        self.final_slot = Some(slot);
        Ok(FixpointSummary {
            state,
            full_provenance: last_iteration.full_provenance,
            last_iteration,
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
        Self::with_seed(seed_rows, seed_rows)
    }

    pub fn with_seed(full_rows: usize, delta_rows: usize) -> Self {
        Self {
            full_rows,
            delta_rows,
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

    fn forced_gpu_policies() -> IterationPolicies {
        IterationPolicies {
            join: JoinPlacementPolicy {
                gpu_min_delta_rows: 0,
                gpu_unavailable_parallel_min_rows: usize::MAX,
            },
            distinct: DistinctPlacementPolicy {
                cpu_produced_gpu_min_rows: 0,
                gpu_produced_gpu_min_rows: 0,
                gpu_unavailable_parallel_min_rows: usize::MAX,
            },
            anti_join: AntiJoinPlacementPolicy {
                cpu_produced_gpu_min_rows: 0,
                gpu_produced_gpu_min_rows: 0,
                cpu_produced_parallel_min_rows: usize::MAX,
                gpu_produced_parallel_min_rows: usize::MAX,
            },
            union: UnionPlacementPolicy {
                cpu_produced_gpu_min_rows: 0,
                gpu_produced_gpu_min_rows: 0,
                gpu_unavailable_parallel_min_rows: usize::MAX,
            },
        }
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

    #[test]
    fn fixpoint_driver_converges_with_cpu_and_cuda_execution() {
        let edge = relation2(&[(1, 2), (2, 3), (3, 4), (4, 5)]);
        let index = U32RangeIndex::build(edge.column(0).unwrap()).unwrap();
        let expected_first = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4];
        let expected_second = [2, 3, 4, 5, 3, 4, 5, 4, 5, 5];

        let mut cpu = FixpointDriver::new().unwrap();
        let cpu_summary = cpu
            .run(
                edge.view(),
                edge.view(),
                edge.view(),
                &index,
                InputProvenance::Cpu,
                None,
                IterationPolicies::default(),
                16,
            )
            .unwrap();
        assert_eq!(cpu_summary.state.iterations(), 4);
        assert_eq!(cpu_summary.state.full_rows(), 10);
        assert!(cpu_summary.state.reached_fixpoint());
        assert!(cpu.newt().unwrap().is_empty());
        assert_eq!(
            cpu.full().unwrap().view().column_slice(0).unwrap(),
            expected_first
        );
        assert_eq!(
            cpu.full().unwrap().view().column_slice(1).unwrap(),
            expected_second
        );

        let stream = CudaStream::new().unwrap();
        let mut gpu = FixpointDriver::new().unwrap();
        let gpu_summary = gpu
            .run(
                edge.view(),
                edge.view(),
                edge.view(),
                &index,
                InputProvenance::Cpu,
                Some(&stream),
                forced_gpu_policies(),
                16,
            )
            .unwrap();
        assert_eq!(gpu_summary.state, cpu_summary.state);
        assert_eq!(gpu_summary.full_provenance, InputProvenance::Gpu);
        assert_eq!(
            gpu.full().unwrap().view().column_slice(0),
            cpu.full().unwrap().view().column_slice(0)
        );
        assert_eq!(
            gpu.full().unwrap().view().column_slice(1),
            cpu.full().unwrap().view().column_slice(1)
        );
    }

    #[test]
    fn fixpoint_driver_reports_iteration_limit() {
        let edge = relation2(&[(1, 2), (2, 3), (3, 4), (4, 5)]);
        let index = U32RangeIndex::build(edge.column(0).unwrap()).unwrap();
        let mut driver = FixpointDriver::new().unwrap();

        let error = driver
            .run(
                edge.view(),
                edge.view(),
                edge.view(),
                &index,
                InputProvenance::Cpu,
                None,
                IterationPolicies::default(),
                2,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            FixpointError::IterationLimit {
                limit: 2,
                remaining_delta_rows: 2
            }
        ));
    }
}
