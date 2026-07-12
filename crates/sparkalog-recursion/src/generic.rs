use crate::{IterationPlacements, IterationPolicies, combine_provenance, placement_provenance};
use sparkalog_execution::{
    CudaStream, InputProvenance, Placement, anti_join_auto, distinct_auto, join_auto, union_auto,
};
use sparkalog_relational::{
    BinaryDistinct, BinaryEqualityJoin, JoinInput, JoinProjection, RecursiveRulePlan,
    RecursiveSccPlan, RelationId, SortedBinaryAntiJoin, SortedBinaryUnion,
};
use sparkalog_storage::{
    AntiJoinWorkspace, DistinctWorkspace, JoinWorkspace, RelationBuffer, RelationView,
    U32RangeIndex, UnionWorkspace,
};

pub struct RelationStore {
    relations: Vec<Option<RelationState>>,
}

struct RelationState {
    full: RelationBuffer,
    delta: Option<RelationBuffer>,
    newt: Option<RelationBuffer>,
    full_provenance: InputProvenance,
    delta_provenance: InputProvenance,
    version: u64,
}

impl RelationStore {
    pub fn new() -> Self {
        Self {
            relations: Vec::new(),
        }
    }

    pub fn insert_static(
        &mut self,
        id: RelationId,
        relation: RelationView<'_>,
        provenance: InputProvenance,
    ) -> Result<(), RelationStoreError> {
        self.insert(
            id,
            RelationState {
                full: RelationBuffer::from_view(relation)?,
                delta: None,
                newt: None,
                full_provenance: provenance,
                delta_provenance: provenance,
                version: 0,
            },
        )
    }

    pub fn insert_recursive(
        &mut self,
        id: RelationId,
        full: RelationView<'_>,
        delta: RelationView<'_>,
        provenance: InputProvenance,
    ) -> Result<(), RelationStoreError> {
        if full.arity() != delta.arity() {
            return Err(RelationStoreError::ArityMismatch {
                full: full.arity(),
                delta: delta.arity(),
            });
        }
        self.insert(
            id,
            RelationState {
                full: RelationBuffer::from_view(full)?,
                delta: Some(RelationBuffer::from_view(delta)?),
                newt: Some(RelationBuffer::with_capacity(full.arity(), 0)?),
                full_provenance: provenance,
                delta_provenance: provenance,
                version: 0,
            },
        )
    }

    pub fn full(&self, id: RelationId) -> Result<&RelationBuffer, RelationStoreError> {
        Ok(&self.state(id)?.full)
    }

    pub fn delta(&self, id: RelationId) -> Result<&RelationBuffer, RelationStoreError> {
        self.state(id)?
            .delta
            .as_ref()
            .ok_or(RelationStoreError::NotRecursive(id))
    }

    pub fn full_provenance(&self, id: RelationId) -> Result<InputProvenance, RelationStoreError> {
        Ok(self.state(id)?.full_provenance)
    }

    fn insert(&mut self, id: RelationId, state: RelationState) -> Result<(), RelationStoreError> {
        let index = id.0 as usize;
        if self.relations.len() <= index {
            self.relations.resize_with(index + 1, || None);
        }
        if self.relations[index].is_some() {
            return Err(RelationStoreError::Duplicate(id));
        }
        self.relations[index] = Some(state);
        Ok(())
    }

    fn state(&self, id: RelationId) -> Result<&RelationState, RelationStoreError> {
        self.relations
            .get(id.0 as usize)
            .and_then(Option::as_ref)
            .ok_or(RelationStoreError::Missing(id))
    }

    fn state_mut(&mut self, id: RelationId) -> Result<&mut RelationState, RelationStoreError> {
        self.relations
            .get_mut(id.0 as usize)
            .and_then(Option::as_mut)
            .ok_or(RelationStoreError::Missing(id))
    }
}

impl Default for RelationStore {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum RelationStoreError {
    Missing(RelationId),
    Duplicate(RelationId),
    NotRecursive(RelationId),
    ArityMismatch { full: usize, delta: usize },
    Storage(sparkalog_storage::Error),
}

impl std::fmt::Display for RelationStoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(id) => write!(formatter, "relation {} is not registered", id.0),
            Self::Duplicate(id) => write!(formatter, "relation {} is already registered", id.0),
            Self::NotRecursive(id) => {
                write!(formatter, "relation {} has no DELTA/NEWT state", id.0)
            }
            Self::ArityMismatch { full, delta } => {
                write!(
                    formatter,
                    "FULL arity {full} differs from DELTA arity {delta}"
                )
            }
            Self::Storage(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RelationStoreError {}

impl From<sparkalog_storage::Error> for RelationStoreError {
    fn from(error: sparkalog_storage::Error) -> Self {
        Self::Storage(error)
    }
}

struct CachedIndex {
    relation: RelationId,
    version: u64,
    column: u32,
    index: U32RangeIndex,
}

#[derive(Debug, Clone, Copy)]
struct PendingRule {
    placements: IterationPlacements,
    newt_provenance: InputProvenance,
}

struct RuleRuntime {
    plan: RecursiveRulePlan,
    index: Option<CachedIndex>,
    candidates: JoinWorkspace,
    distinct: DistinctWorkspace,
    newt: AntiJoinWorkspace,
    union: UnionWorkspace,
    pending: Option<PendingRule>,
}

impl RuleRuntime {
    fn new(plan: RecursiveRulePlan) -> Result<Self, sparkalog_storage::Error> {
        Ok(Self {
            plan,
            index: None,
            candidates: JoinWorkspace::new(2)?,
            distinct: DistinctWorkspace::new()?,
            newt: AntiJoinWorkspace::new()?,
            union: UnionWorkspace::new()?,
            pending: None,
        })
    }

    fn evaluate(
        &mut self,
        store: &RelationStore,
        stream: Option<&CudaStream>,
        policies: IterationPolicies,
    ) -> Result<(), GenericExecutionError> {
        let delta_state = store.state(self.plan.delta_input)?;
        let right_state = store.state(self.plan.right_input)?;
        let target_state = store.state(self.plan.target)?;
        let delta = delta_state
            .delta
            .as_ref()
            .ok_or(RelationStoreError::NotRecursive(self.plan.delta_input))?;
        let right_column = right_state
            .full
            .view()
            .column_slice(self.plan.join.right_key as usize)
            .ok_or(GenericExecutionError::MissingPlanColumn {
                relation: self.plan.right_input,
                column: self.plan.join.right_key,
            })?;
        let rebuild_index = self.index.as_ref().is_none_or(|cached| {
            cached.relation != self.plan.right_input
                || cached.version != right_state.version
                || cached.column != self.plan.join.right_key
        });
        if rebuild_index {
            self.index = Some(CachedIndex {
                relation: self.plan.right_input,
                version: right_state.version,
                column: self.plan.join.right_key,
                index: U32RangeIndex::build_slice(right_column)?,
            });
        }
        let index = &self.index.as_ref().expect("index was populated").index;
        let join = join_auto(
            delta.view(),
            right_state.full.view(),
            index,
            self.plan.join,
            &mut self.candidates,
            stream,
            policies.join,
        )?;
        let join_provenance = placement_provenance(join);
        let distinct = distinct_auto(
            self.candidates.output().view(),
            self.plan.distinct,
            join_provenance,
            &mut self.distinct,
            stream,
            policies.distinct,
        )?;
        let distinct_provenance = placement_provenance(distinct);
        let anti_provenance = combine_provenance(distinct_provenance, target_state.full_provenance);
        let anti_join = anti_join_auto(
            self.distinct.output().view(),
            target_state.full.view(),
            self.plan.anti_join,
            anti_provenance,
            &mut self.newt,
            stream,
            policies.anti_join,
        )?;
        self.pending = Some(PendingRule {
            placements: IterationPlacements {
                join,
                distinct,
                anti_join,
                union: Placement::CpuSerial,
            },
            newt_provenance: placement_provenance(anti_join),
        });
        Ok(())
    }

    fn apply(
        &mut self,
        store: &mut RelationStore,
        stream: Option<&CudaStream>,
        policies: IterationPolicies,
    ) -> Result<PendingRule, GenericExecutionError> {
        let mut pending = self
            .pending
            .take()
            .expect("rule was evaluated before apply");
        let target = store.state_mut(self.plan.target)?;
        let union_provenance = combine_provenance(target.full_provenance, pending.newt_provenance);
        let union = union_auto(
            target.full.view(),
            self.newt.output().view(),
            self.plan.union,
            union_provenance,
            &mut self.union,
            stream,
            policies.union,
        )?;
        let delta = target
            .delta
            .as_mut()
            .ok_or(RelationStoreError::NotRecursive(self.plan.target))?;
        let newt = target
            .newt
            .as_mut()
            .ok_or(RelationStoreError::NotRecursive(self.plan.target))?;
        self.newt.swap_output(newt);
        self.union.swap_output(&mut target.full);
        std::mem::swap(delta, newt);
        target.delta_provenance = pending.newt_provenance;
        target.full_provenance = placement_provenance(union);
        target.version = target.version.wrapping_add(1);
        pending.placements.union = union;
        Ok(pending)
    }
}

pub struct RecursiveExecutor {
    relations: Vec<RelationId>,
    rules: Vec<RuleRuntime>,
}

impl RecursiveExecutor {
    pub fn compile(
        plan: RecursiveSccPlan,
        store: &RelationStore,
    ) -> Result<Self, GenericExecutionError> {
        if plan.relations.is_empty() {
            return Err(GenericExecutionError::EmptyScc);
        }
        let mut targets = Vec::with_capacity(plan.rules.len());
        for &relation in &plan.relations {
            let state = store.state(relation)?;
            if state.delta.is_none() {
                return Err(RelationStoreError::NotRecursive(relation).into());
            }
        }
        for rule in &plan.rules {
            store.state(rule.delta_input)?;
            store.state(rule.right_input)?;
            if !plan.relations.contains(&rule.target) {
                return Err(GenericExecutionError::TargetOutsideScc(rule.target));
            }
            if targets.contains(&rule.target) {
                return Err(GenericExecutionError::DuplicateRuleTarget(rule.target));
            }
            targets.push(rule.target);
        }
        for &relation in &plan.relations {
            if !targets.contains(&relation) {
                return Err(GenericExecutionError::MissingRuleTarget(relation));
            }
        }
        let rules = plan
            .rules
            .into_iter()
            .map(RuleRuntime::new)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            relations: plan.relations,
            rules,
        })
    }

    pub fn run(
        &mut self,
        store: &mut RelationStore,
        stream: Option<&CudaStream>,
        policies: IterationPolicies,
        max_iterations: usize,
    ) -> Result<SccSummary, GenericExecutionError> {
        let mut iterations = 0;
        let mut total_new_rows = 0;
        let mut placements = Vec::new();
        while self.has_delta(store)? {
            if iterations >= max_iterations {
                return Err(GenericExecutionError::IterationLimit {
                    limit: max_iterations,
                    remaining_delta_rows: self.delta_rows(store)?,
                });
            }
            for rule in &mut self.rules {
                rule.evaluate(store, stream, policies)?;
            }
            placements.clear();
            for rule in &mut self.rules {
                let target = rule.plan.target;
                let pending = rule.apply(store, stream, policies)?;
                let new_rows = store.delta(target)?.len();
                total_new_rows += new_rows;
                placements.push((target, pending.placements));
            }
            iterations += 1;
        }
        let relation_rows = self
            .relations
            .iter()
            .map(|&id| Ok((id, store.full(id)?.len())))
            .collect::<Result<Vec<_>, RelationStoreError>>()?;
        Ok(SccSummary {
            iterations,
            total_new_rows,
            relation_rows,
            last_placements: placements,
        })
    }

    fn has_delta(&self, store: &RelationStore) -> Result<bool, RelationStoreError> {
        for &relation in &self.relations {
            if !store.delta(relation)?.is_empty() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn delta_rows(&self, store: &RelationStore) -> Result<usize, RelationStoreError> {
        let mut rows = 0_usize;
        for &relation in &self.relations {
            rows = rows.saturating_add(store.delta(relation)?.len());
        }
        Ok(rows)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SccSummary {
    pub iterations: usize,
    pub total_new_rows: usize,
    pub relation_rows: Vec<(RelationId, usize)>,
    pub last_placements: Vec<(RelationId, IterationPlacements)>,
}

#[derive(Debug)]
pub enum GenericExecutionError {
    Store(RelationStoreError),
    Storage(sparkalog_storage::Error),
    Execution(sparkalog_execution::Error),
    EmptyScc,
    TargetOutsideScc(RelationId),
    DuplicateRuleTarget(RelationId),
    MissingRuleTarget(RelationId),
    MissingPlanColumn {
        relation: RelationId,
        column: u32,
    },
    IterationLimit {
        limit: usize,
        remaining_delta_rows: usize,
    },
}

impl std::fmt::Display for GenericExecutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(error) => error.fmt(formatter),
            Self::Storage(error) => error.fmt(formatter),
            Self::Execution(error) => error.fmt(formatter),
            Self::EmptyScc => formatter.write_str("recursive SCC has no relations"),
            Self::TargetOutsideScc(id) => {
                write!(formatter, "rule target {} is outside its SCC", id.0)
            }
            Self::DuplicateRuleTarget(id) => {
                write!(formatter, "relation {} has multiple rules", id.0)
            }
            Self::MissingRuleTarget(id) => {
                write!(formatter, "relation {} has no recursive rule", id.0)
            }
            Self::MissingPlanColumn { relation, column } => {
                write!(
                    formatter,
                    "relation {} has no planned column {column}",
                    relation.0
                )
            }
            Self::IterationLimit {
                limit,
                remaining_delta_rows,
            } => write!(
                formatter,
                "recursive SCC did not converge within {limit} iterations; {remaining_delta_rows} delta rows remain"
            ),
        }
    }
}

impl std::error::Error for GenericExecutionError {}

impl From<RelationStoreError> for GenericExecutionError {
    fn from(error: RelationStoreError) -> Self {
        Self::Store(error)
    }
}

impl From<sparkalog_storage::Error> for GenericExecutionError {
    fn from(error: sparkalog_storage::Error) -> Self {
        Self::Storage(error)
    }
}

impl From<sparkalog_execution::Error> for GenericExecutionError {
    fn from(error: sparkalog_execution::Error) -> Self {
        Self::Execution(error)
    }
}

pub fn transitive_closure_scc(path: RelationId, edge: RelationId) -> RecursiveSccPlan {
    RecursiveSccPlan {
        relations: vec![path],
        rules: vec![RecursiveRulePlan {
            target: path,
            delta_input: path,
            right_input: edge,
            join: BinaryEqualityJoin {
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
            },
            distinct: BinaryDistinct { columns: [0, 1] },
            anti_join: SortedBinaryAntiJoin {
                left: [0, 1],
                right: [0, 1],
            },
            union: SortedBinaryUnion {
                left: [0, 1],
                right: [0, 1],
            },
        }],
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

    fn recursive_rule(
        target: RelationId,
        delta_input: RelationId,
        right_input: RelationId,
    ) -> RecursiveRulePlan {
        let mut plan = transitive_closure_scc(target, right_input)
            .rules
            .pop()
            .unwrap();
        plan.delta_input = delta_input;
        plan
    }

    fn forced_gpu_policies() -> IterationPolicies {
        IterationPolicies {
            join: sparkalog_execution::JoinPlacementPolicy {
                gpu_min_delta_rows: 0,
                gpu_unavailable_parallel_min_rows: usize::MAX,
            },
            distinct: sparkalog_execution::DistinctPlacementPolicy {
                cpu_produced_gpu_min_rows: 0,
                gpu_produced_gpu_min_rows: 0,
                gpu_unavailable_parallel_min_rows: usize::MAX,
            },
            anti_join: sparkalog_execution::AntiJoinPlacementPolicy {
                cpu_produced_gpu_min_rows: 0,
                gpu_produced_gpu_min_rows: 0,
                cpu_produced_parallel_min_rows: usize::MAX,
                gpu_produced_parallel_min_rows: usize::MAX,
            },
            union: sparkalog_execution::UnionPlacementPolicy {
                cpu_produced_gpu_min_rows: 0,
                gpu_produced_gpu_min_rows: 0,
                gpu_unavailable_parallel_min_rows: usize::MAX,
            },
        }
    }

    #[test]
    fn generic_plan_reaches_transitive_closure() {
        let edge_id = RelationId(0);
        let path_id = RelationId(1);
        let edge = relation2(&[(1, 2), (2, 3), (3, 4), (4, 5)]);
        let mut store = RelationStore::new();
        store
            .insert_static(edge_id, edge.view(), InputProvenance::Cpu)
            .unwrap();
        store
            .insert_recursive(path_id, edge.view(), edge.view(), InputProvenance::Cpu)
            .unwrap();
        let mut executor =
            RecursiveExecutor::compile(transitive_closure_scc(path_id, edge_id), &store).unwrap();

        let summary = executor
            .run(&mut store, None, IterationPolicies::default(), 16)
            .unwrap();

        assert_eq!(summary.iterations, 4);
        assert_eq!(summary.total_new_rows, 6);
        assert_eq!(summary.relation_rows, [(path_id, 10)]);
        assert!(store.delta(path_id).unwrap().is_empty());
        assert_eq!(
            store.full(path_id).unwrap().view().column_slice(0).unwrap(),
            &[1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
        );
        assert_eq!(
            store.full(path_id).unwrap().view().column_slice(1).unwrap(),
            &[2, 3, 4, 5, 3, 4, 5, 4, 5, 5]
        );
    }

    #[test]
    fn generic_plan_reaches_the_same_closure_on_cuda() {
        let edge_id = RelationId(0);
        let path_id = RelationId(1);
        let edge = relation2(&[(1, 2), (2, 3), (3, 4)]);
        let mut store = RelationStore::new();
        store
            .insert_static(edge_id, edge.view(), InputProvenance::Cpu)
            .unwrap();
        store
            .insert_recursive(path_id, edge.view(), edge.view(), InputProvenance::Cpu)
            .unwrap();
        let mut executor =
            RecursiveExecutor::compile(transitive_closure_scc(path_id, edge_id), &store).unwrap();
        let stream = CudaStream::new().unwrap();

        let summary = executor
            .run(&mut store, Some(&stream), forced_gpu_policies(), 16)
            .unwrap();

        assert_eq!(summary.relation_rows, [(path_id, 6)]);
        assert!(
            summary
                .last_placements
                .iter()
                .all(|(_, placements)| placements.join == Placement::Gpu
                    && placements.distinct == Placement::Gpu
                    && placements.anti_join == Placement::Gpu
                    && placements.union == Placement::Gpu)
        );
        assert_eq!(
            store.full(path_id).unwrap().view().column_slice(0).unwrap(),
            &[1, 1, 1, 2, 2, 3]
        );
        assert_eq!(
            store.full(path_id).unwrap().view().column_slice(1).unwrap(),
            &[2, 3, 4, 3, 4, 4]
        );
    }

    #[test]
    fn mutually_recursive_rules_observe_the_same_round() {
        let edge_id = RelationId(0);
        let a_id = RelationId(1);
        let b_id = RelationId(2);
        let edge = relation2(&[(2, 3), (3, 4), (4, 5)]);
        let a_seed = relation2(&[(1, 2)]);
        let b_seed = relation2(&[(2, 3)]);
        let mut store = RelationStore::new();
        store
            .insert_static(edge_id, edge.view(), InputProvenance::Cpu)
            .unwrap();
        store
            .insert_recursive(a_id, a_seed.view(), a_seed.view(), InputProvenance::Cpu)
            .unwrap();
        store
            .insert_recursive(b_id, b_seed.view(), b_seed.view(), InputProvenance::Cpu)
            .unwrap();
        let plan = RecursiveSccPlan {
            relations: vec![a_id, b_id],
            rules: vec![
                recursive_rule(a_id, b_id, edge_id),
                recursive_rule(b_id, a_id, edge_id),
            ],
        };
        let mut executor = RecursiveExecutor::compile(plan, &store).unwrap();

        let summary = executor
            .run(&mut store, None, IterationPolicies::default(), 16)
            .unwrap();

        assert_eq!(summary.iterations, 4);
        assert!(store.delta(a_id).unwrap().is_empty());
        assert!(store.delta(b_id).unwrap().is_empty());
        assert_eq!(
            store.full(a_id).unwrap().view().column_slice(0).unwrap(),
            &[1, 1, 2]
        );
        assert_eq!(
            store.full(a_id).unwrap().view().column_slice(1).unwrap(),
            &[2, 4, 4]
        );
        assert_eq!(
            store.full(b_id).unwrap().view().column_slice(0).unwrap(),
            &[1, 1, 2, 2]
        );
        assert_eq!(
            store.full(b_id).unwrap().view().column_slice(1).unwrap(),
            &[3, 5, 3, 5]
        );
    }
}
