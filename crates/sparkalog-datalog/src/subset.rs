use std::collections::{BTreeSet, HashSet};

use sparkalog_execution::{CudaStream, InputProvenance};
use sparkalog_recursion::{
    GenericExecutionError, IterationPolicies, RecursiveExecutor, RelationStore, SccSummary,
};
use sparkalog_relational::{
    BinaryDistinct, BinaryEqualityJoin, JoinInput, JoinProjection, RecursiveRulePlan,
    RecursiveSccPlan, RelationId, SortedBinaryAntiJoin, SortedBinaryUnion,
};
use sparkalog_storage::Relation;

use crate::{
    PredicateId, ResolvedAtom, ResolvedProgram, ResolvedRule, ResolvedTerm, ScheduledScc, Span,
    VariableId, stratify,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinarySeedPlan {
    pub target: RelationId,
    pub source: RelationId,
    pub columns: [u32; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinarySccPlan {
    pub plan: RecursiveSccPlan,
    pub seeds: Vec<BinarySeedPlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryProgramPlan {
    pub predicate_count: usize,
    pub facts: Vec<(RelationId, [u32; 2])>,
    pub recursive_sccs: Vec<BinarySccPlan>,
    pub outputs: Vec<RelationId>,
}

pub fn lower_binary(program: &ResolvedProgram) -> Result<BinaryProgramPlan, LoweringError> {
    let schedule = stratify(program).map_err(|error| LoweringError {
        message: error.to_string(),
        span: Span::default(),
    })?;
    validate_binary_arities(program)?;
    let recursive_sccs = schedule
        .strata
        .iter()
        .flat_map(|stratum| stratum.sccs.iter())
        .filter(|scc| scc.recursive)
        .cloned()
        .collect::<Vec<_>>();
    let recursive_predicates = recursive_sccs
        .iter()
        .flat_map(|scc| scc.predicates.iter().copied())
        .collect::<HashSet<_>>();
    for rule in program.rules.iter().filter(|rule| !rule.is_fact()) {
        if !recursive_predicates.contains(&rule.head.predicate) {
            return Err(lowering_error(
                rule.span,
                "the initial executable subset only derives recursive predicates",
            ));
        }
    }

    let mut lowered_sccs = Vec::with_capacity(recursive_sccs.len());
    for scc in &recursive_sccs {
        lowered_sccs.push(lower_scc(program, scc)?);
    }
    let facts = program
        .rules
        .iter()
        .filter(|rule| rule.is_fact())
        .map(lower_fact)
        .collect::<Result<Vec<_>, _>>()?;
    let predicate_count = program
        .rules
        .iter()
        .flat_map(|rule| {
            std::iter::once(rule.head.predicate)
                .chain(rule.body.iter().map(|literal| literal.atom.predicate))
        })
        .chain(program.declarations.iter().copied())
        .chain(program.outputs.iter().copied())
        .map(|id| id.0 as usize + 1)
        .max()
        .unwrap_or(0);
    Ok(BinaryProgramPlan {
        predicate_count,
        facts,
        recursive_sccs: lowered_sccs,
        outputs: program
            .outputs
            .iter()
            .map(|predicate| relation_id(*predicate))
            .collect(),
    })
}

fn validate_binary_arities(program: &ResolvedProgram) -> Result<(), LoweringError> {
    for rule in &program.rules {
        if rule.head.terms.len() != 2 {
            return Err(lowering_error(
                rule.head.span,
                "the initial executable subset requires binary predicates",
            ));
        }
        for literal in &rule.body {
            if literal.atom.terms.len() != 2 {
                return Err(lowering_error(
                    literal.atom.span,
                    "the initial executable subset requires binary predicates",
                ));
            }
        }
    }
    Ok(())
}

fn lower_scc(
    program: &ResolvedProgram,
    scc: &ScheduledScc,
) -> Result<BinarySccPlan, LoweringError> {
    let predicates = scc.predicates.iter().copied().collect::<HashSet<_>>();
    let mut seeds = Vec::new();
    let mut rules = Vec::new();
    for rule in program
        .rules
        .iter()
        .filter(|rule| predicates.contains(&rule.head.predicate) && !rule.is_fact())
    {
        let recursive_literals = rule
            .body
            .iter()
            .filter(|literal| predicates.contains(&literal.atom.predicate))
            .count();
        match recursive_literals {
            0 => seeds.push(lower_seed(rule)?),
            1 => rules.push(lower_recursive_rule(rule, &predicates)?),
            _ => {
                return Err(lowering_error(
                    rule.span,
                    "the initial executable subset permits one recursive body atom",
                ));
            }
        }
    }
    Ok(BinarySccPlan {
        plan: RecursiveSccPlan {
            relations: scc
                .predicates
                .iter()
                .map(|predicate| relation_id(*predicate))
                .collect(),
            rules,
        },
        seeds,
    })
}

fn lower_fact(rule: &ResolvedRule) -> Result<(RelationId, [u32; 2]), LoweringError> {
    let values = rule
        .head
        .terms
        .iter()
        .map(|term| match term {
            ResolvedTerm::Value(value) => Ok(value.0),
            ResolvedTerm::Variable(_) => Err(lowering_error(
                rule.head.span,
                "facts cannot contain variables",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((relation_id(rule.head.predicate), [values[0], values[1]]))
}

fn lower_seed(rule: &ResolvedRule) -> Result<BinarySeedPlan, LoweringError> {
    if rule.body.len() != 1 || rule.body[0].negated {
        return Err(lowering_error(
            rule.span,
            "a seed clause must contain one positive atom",
        ));
    }
    let source = &rule.body[0].atom;
    let columns = projection_columns(&rule.head, source)?;
    Ok(BinarySeedPlan {
        target: relation_id(rule.head.predicate),
        source: relation_id(source.predicate),
        columns,
    })
}

fn lower_recursive_rule(
    rule: &ResolvedRule,
    scc: &HashSet<PredicateId>,
) -> Result<RecursiveRulePlan, LoweringError> {
    if rule.body.len() != 2 || rule.body.iter().any(|literal| literal.negated) {
        return Err(lowering_error(
            rule.span,
            "a recursive clause must contain two positive atoms",
        ));
    }
    let recursive = rule
        .body
        .iter()
        .find(|literal| scc.contains(&literal.atom.predicate))
        .expect("recursive literal count was checked");
    let right = rule
        .body
        .iter()
        .find(|literal| !scc.contains(&literal.atom.predicate))
        .ok_or_else(|| {
            lowering_error(
                rule.span,
                "the initial subset requires a non-recursive right input",
            )
        })?;
    let (left_key, right_key) = single_join_key(&recursive.atom, &right.atom)?;
    let output = head_projection(&rule.head, &recursive.atom, &right.atom)?;
    Ok(RecursiveRulePlan {
        target: relation_id(rule.head.predicate),
        delta_input: relation_id(recursive.atom.predicate),
        right_input: relation_id(right.atom.predicate),
        join: BinaryEqualityJoin {
            left_key,
            right_key,
            output,
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
    })
}

fn projection_columns(
    head: &ResolvedAtom,
    source: &ResolvedAtom,
) -> Result<[u32; 2], LoweringError> {
    let mut columns = [0_u32; 2];
    for (output, term) in head.terms.iter().enumerate() {
        let ResolvedTerm::Variable(variable) = term else {
            return Err(lowering_error(
                head.span,
                "seed clause heads must project variables",
            ));
        };
        columns[output] = variable_column(source, *variable).ok_or_else(|| {
            lowering_error(head.span, "head variable is absent from the seed atom")
        })?;
    }
    Ok(columns)
}

fn single_join_key(left: &ResolvedAtom, right: &ResolvedAtom) -> Result<(u32, u32), LoweringError> {
    let mut keys = Vec::new();
    for (left_column, left_term) in left.terms.iter().enumerate() {
        let ResolvedTerm::Variable(left_variable) = left_term else {
            return Err(lowering_error(
                left.span,
                "recursive joins do not yet support constants",
            ));
        };
        for (right_column, right_term) in right.terms.iter().enumerate() {
            let ResolvedTerm::Variable(right_variable) = right_term else {
                return Err(lowering_error(
                    right.span,
                    "recursive joins do not yet support constants",
                ));
            };
            if left_variable == right_variable {
                keys.push((left_column as u32, right_column as u32));
            }
        }
    }
    if keys.len() == 1 {
        Ok(keys[0])
    } else {
        Err(lowering_error(
            left.span.join(right.span),
            "recursive atoms must share exactly one join variable",
        ))
    }
}

fn head_projection(
    head: &ResolvedAtom,
    left: &ResolvedAtom,
    right: &ResolvedAtom,
) -> Result<[JoinProjection; 2], LoweringError> {
    let mut result = [
        JoinProjection {
            input: JoinInput::Left,
            column: 0,
        },
        JoinProjection {
            input: JoinInput::Left,
            column: 0,
        },
    ];
    for (index, term) in head.terms.iter().enumerate() {
        let ResolvedTerm::Variable(variable) = term else {
            return Err(lowering_error(
                head.span,
                "recursive rule heads must project variables",
            ));
        };
        result[index] = if let Some(column) = variable_column(left, *variable) {
            JoinProjection {
                input: JoinInput::Left,
                column,
            }
        } else if let Some(column) = variable_column(right, *variable) {
            JoinProjection {
                input: JoinInput::Right,
                column,
            }
        } else {
            return Err(lowering_error(
                head.span,
                "head variable is absent from the recursive body",
            ));
        };
    }
    Ok(result)
}

fn variable_column(atom: &ResolvedAtom, variable: VariableId) -> Option<u32> {
    atom.terms
        .iter()
        .position(|term| *term == ResolvedTerm::Variable(variable))
        .map(|column| column as u32)
}

fn relation_id(predicate: PredicateId) -> RelationId {
    RelationId(predicate.0)
}

fn lowering_error(span: Span, message: impl Into<String>) -> LoweringError {
    LoweringError {
        message: message.into(),
        span,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweringError {
    pub message: String,
    pub span: Span,
}

impl std::fmt::Display for LoweringError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(formatter)
    }
}

impl std::error::Error for LoweringError {}

pub struct BinaryExecution {
    pub store: RelationStore,
    pub summaries: Vec<SccSummary>,
}

pub fn execute_binary(
    plan: &BinaryProgramPlan,
    stream: Option<&CudaStream>,
    policies: IterationPolicies,
    max_iterations: usize,
) -> Result<BinaryExecution, BinaryExecutionError> {
    let recursive = plan
        .recursive_sccs
        .iter()
        .flat_map(|scc| scc.plan.relations.iter().copied())
        .collect::<HashSet<_>>();
    let mut rows = vec![BTreeSet::<[u32; 2]>::new(); plan.predicate_count];
    for &(relation, tuple) in &plan.facts {
        rows[relation.0 as usize].insert(tuple);
    }
    for scc in &plan.recursive_sccs {
        for seed in &scc.seeds {
            let source = rows[seed.source.0 as usize].clone();
            for tuple in source {
                rows[seed.target.0 as usize].insert([
                    tuple[seed.columns[0] as usize],
                    tuple[seed.columns[1] as usize],
                ]);
            }
        }
    }
    let relations = rows
        .iter()
        .map(relation_from_rows)
        .collect::<Result<Vec<_>, _>>()?;
    let mut store = RelationStore::new();
    for (index, relation) in relations.iter().enumerate() {
        let id = RelationId(index as u32);
        if recursive.contains(&id) {
            store.insert_recursive(id, relation.view(), relation.view(), InputProvenance::Cpu)?;
        } else {
            store.insert_static(id, relation.view(), InputProvenance::Cpu)?;
        }
    }
    let mut summaries = Vec::with_capacity(plan.recursive_sccs.len());
    for scc in &plan.recursive_sccs {
        let mut executor = RecursiveExecutor::compile(scc.plan.clone(), &store)?;
        summaries.push(executor.run(&mut store, stream, policies, max_iterations)?);
    }
    Ok(BinaryExecution { store, summaries })
}

fn relation_from_rows(rows: &BTreeSet<[u32; 2]>) -> Result<Relation, sparkalog_storage::Error> {
    let mut relation = Relation::new(2, rows.len())?;
    for (row, tuple) in rows.iter().enumerate() {
        relation
            .column_mut(0)
            .expect("binary relation")
            .as_mut_slice()[row] = tuple[0];
        relation
            .column_mut(1)
            .expect("binary relation")
            .as_mut_slice()[row] = tuple[1];
    }
    Ok(relation)
}

#[derive(Debug)]
pub enum BinaryExecutionError {
    Storage(sparkalog_storage::Error),
    Store(sparkalog_recursion::RelationStoreError),
    Recursion(GenericExecutionError),
}

impl std::fmt::Display for BinaryExecutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Storage(error) => error.fmt(formatter),
            Self::Store(error) => error.fmt(formatter),
            Self::Recursion(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for BinaryExecutionError {}

impl From<sparkalog_storage::Error> for BinaryExecutionError {
    fn from(error: sparkalog_storage::Error) -> Self {
        Self::Storage(error)
    }
}

impl From<sparkalog_recursion::RelationStoreError> for BinaryExecutionError {
    fn from(error: sparkalog_recursion::RelationStoreError) -> Self {
        Self::Store(error)
    }
}

impl From<GenericExecutionError> for BinaryExecutionError {
    fn from(error: GenericExecutionError) -> Self {
        Self::Recursion(error)
    }
}

#[cfg(test)]
mod tests {
    use crate::{ProgramCatalog, parse_program, resolve_program};

    use super::*;

    #[test]
    fn source_program_reaches_transitive_closure() {
        let source = "
            edge(1, 2). edge(2, 3). edge(3, 4).
            path(x, y) :- edge(x, y).
            path(x, z) :- path(x, y), edge(y, z).
            .output path
        ";
        let parsed = parse_program(source);
        assert_eq!(parsed.diagnostics, []);
        let mut catalog = ProgramCatalog::new();
        let resolved = resolve_program(&parsed.program, &mut catalog);
        assert_eq!(resolved.diagnostics, []);
        let plan = lower_binary(&resolved.program).unwrap();

        let execution = execute_binary(&plan, None, IterationPolicies::default(), 16).unwrap();
        let path = catalog.predicates.id("path").unwrap();
        let relation = execution.store.full(relation_id(path)).unwrap().view();

        assert_eq!(execution.summaries[0].iterations, 3);
        assert_eq!(relation.len(), 6);
        assert_eq!(relation.column_slice(0).unwrap(), &[0, 0, 0, 1, 1, 2]);
        assert_eq!(relation.column_slice(1).unwrap(), &[1, 2, 3, 2, 3, 3]);
    }

    #[test]
    fn unsupported_nonrecursive_derivation_is_explicit() {
        let parsed = parse_program("edge(1, 2). copy(x, y) :- edge(x, y).");
        let mut catalog = ProgramCatalog::new();
        let resolved = resolve_program(&parsed.program, &mut catalog);

        let error = lower_binary(&resolved.program).unwrap_err();

        assert!(error.message.contains("only derives recursive predicates"));
    }
}
