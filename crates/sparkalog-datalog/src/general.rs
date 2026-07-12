use std::collections::{BTreeSet, HashSet};

use sparkalog_relational::{
    BindingId, GeneralProgramPlan, GeneralSccPlan, GeneralStratumPlan, PlanTerm, PlannedAtom,
    RelationId, RelationVersion, RelationalClausePlan,
};

use crate::{
    LoweringError, PredicateId, ResolvedAtom, ResolvedProgram, ResolvedRule, ResolvedTerm,
    ScheduledScc, Span, stratify,
};

type TupleSet = BTreeSet<Vec<u32>>;
type Contributions = Vec<(RelationId, TupleSet)>;

pub fn lower_general(program: &ResolvedProgram) -> Result<GeneralProgramPlan, LoweringError> {
    let schedule = stratify(program).map_err(|error| LoweringError {
        message: error.to_string(),
        span: Span::default(),
    })?;
    let predicate_count = program
        .rules
        .iter()
        .flat_map(|rule| {
            std::iter::once(rule.head.predicate)
                .chain(rule.body.iter().map(|literal| literal.atom.predicate))
        })
        .chain(program.outputs.iter().copied())
        .map(|id| id.0 as usize + 1)
        .max()
        .unwrap_or(0);
    let facts = program
        .rules
        .iter()
        .filter(|rule| rule.is_fact())
        .map(|rule| {
            let values = rule
                .head
                .terms
                .iter()
                .map(|term| match term {
                    ResolvedTerm::Value(value) => Ok(value.0),
                    ResolvedTerm::Variable(_) => Err(LoweringError {
                        message: "facts cannot contain variables".into(),
                        span: rule.head.span,
                    }),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok((relation_id(rule.head.predicate), values))
        })
        .collect::<Result<Vec<_>, LoweringError>>()?;
    let strata = schedule
        .strata
        .iter()
        .map(|stratum| {
            let sccs = stratum
                .sccs
                .iter()
                .map(|scc| lower_scc(program, scc))
                .collect::<Vec<_>>();
            GeneralStratumPlan {
                index: stratum.index,
                sccs,
            }
        })
        .collect();
    Ok(GeneralProgramPlan {
        predicate_count,
        facts,
        strata,
        outputs: program
            .outputs
            .iter()
            .map(|predicate| relation_id(*predicate))
            .collect(),
    })
}

fn lower_scc(program: &ResolvedProgram, scc: &ScheduledScc) -> GeneralSccPlan {
    let predicates = scc.predicates.iter().copied().collect::<HashSet<_>>();
    let mut seeds = Vec::new();
    let mut recursive_variants = Vec::new();
    for rule in program
        .rules
        .iter()
        .filter(|rule| !rule.is_fact() && predicates.contains(&rule.head.predicate))
    {
        let recursive_positions = rule
            .body
            .iter()
            .enumerate()
            .filter_map(|(position, literal)| {
                (!literal.negated && predicates.contains(&literal.atom.predicate))
                    .then_some(position)
            })
            .collect::<Vec<_>>();
        if recursive_positions.is_empty() {
            seeds.push(lower_clause(rule, None));
        } else {
            recursive_variants.extend(
                recursive_positions
                    .into_iter()
                    .map(|position| lower_clause(rule, Some(position))),
            );
        }
    }
    GeneralSccPlan {
        relations: scc
            .predicates
            .iter()
            .map(|predicate| relation_id(*predicate))
            .collect(),
        recursive: scc.recursive,
        seeds,
        recursive_variants,
    }
}

fn lower_clause(rule: &ResolvedRule, delta_position: Option<usize>) -> RelationalClausePlan {
    let mut positive = Vec::new();
    let mut negative = Vec::new();
    for (position, literal) in rule.body.iter().enumerate() {
        let atom = planned_atom(
            &literal.atom,
            if delta_position == Some(position) {
                RelationVersion::Delta
            } else {
                RelationVersion::Full
            },
        );
        if literal.negated {
            negative.push(atom);
        } else {
            positive.push(atom);
        }
    }
    RelationalClausePlan {
        target: relation_id(rule.head.predicate),
        head: rule.head.terms.iter().copied().map(plan_term).collect(),
        positive,
        negative,
    }
}

fn planned_atom(atom: &ResolvedAtom, version: RelationVersion) -> PlannedAtom {
    PlannedAtom {
        relation: relation_id(atom.predicate),
        version,
        terms: atom.terms.iter().copied().map(plan_term).collect(),
    }
}

fn plan_term(term: ResolvedTerm) -> PlanTerm {
    match term {
        ResolvedTerm::Variable(variable) => PlanTerm::Binding(BindingId(variable.0)),
        ResolvedTerm::Value(value) => PlanTerm::Value(value.0),
    }
}

fn relation_id(predicate: PredicateId) -> RelationId {
    RelationId(predicate.0)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TupleStore {
    full: Vec<TupleSet>,
    delta: Vec<TupleSet>,
}

impl TupleStore {
    pub fn rows(&self, relation: RelationId) -> Option<&TupleSet> {
        self.full.get(relation.0 as usize)
    }

    pub fn delta(&self, relation: RelationId) -> Option<&TupleSet> {
        self.delta.get(relation.0 as usize)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneralSccSummary {
    pub relations: Vec<(RelationId, usize)>,
    pub iterations: usize,
    pub new_rows: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneralExecution {
    pub store: TupleStore,
    pub summaries: Vec<GeneralSccSummary>,
}

pub fn execute_general(
    plan: &GeneralProgramPlan,
    max_iterations: usize,
) -> Result<GeneralExecution, GeneralExecutionError> {
    let mut store = TupleStore {
        full: vec![BTreeSet::new(); plan.predicate_count],
        delta: vec![BTreeSet::new(); plan.predicate_count],
    };
    for (relation, tuple) in &plan.facts {
        store.full[relation.0 as usize].insert(tuple.clone());
    }
    let mut summaries = Vec::new();
    for stratum in &plan.strata {
        for scc in &stratum.sccs {
            summaries.push(execute_scc(scc, &mut store, max_iterations)?);
        }
    }
    Ok(GeneralExecution { store, summaries })
}

fn execute_scc(
    scc: &GeneralSccPlan,
    store: &mut TupleStore,
    max_iterations: usize,
) -> Result<GeneralSccSummary, GeneralExecutionError> {
    let seed_contributions = evaluate_clauses(&scc.seeds, store)?;
    for (target, tuples) in seed_contributions {
        store.full[target.0 as usize].extend(tuples);
    }
    if !scc.recursive {
        return Ok(summary(scc, store, usize::from(!scc.seeds.is_empty()), 0));
    }
    for &relation in &scc.relations {
        store.delta[relation.0 as usize] = store.full[relation.0 as usize].clone();
    }
    let mut iterations = 0;
    let mut total_new = 0;
    while scc
        .relations
        .iter()
        .any(|relation| !store.delta[relation.0 as usize].is_empty())
    {
        if iterations >= max_iterations {
            return Err(GeneralExecutionError::IterationLimit {
                limit: max_iterations,
            });
        }
        let contributions = evaluate_clauses(&scc.recursive_variants, store)?;
        let mut next = scc
            .relations
            .iter()
            .map(|&relation| (relation, BTreeSet::new()))
            .collect::<Vec<_>>();
        for (target, tuples) in contributions {
            let target_next = next
                .iter_mut()
                .find(|(relation, _)| *relation == target)
                .expect("clause target belongs to SCC");
            for tuple in tuples {
                if !store.full[target.0 as usize].contains(&tuple) {
                    target_next.1.insert(tuple);
                }
            }
        }
        for (relation, tuples) in next {
            total_new += tuples.len();
            store.full[relation.0 as usize].extend(tuples.iter().cloned());
            store.delta[relation.0 as usize] = tuples;
        }
        iterations += 1;
    }
    Ok(summary(scc, store, iterations, total_new))
}

fn summary(
    scc: &GeneralSccPlan,
    store: &TupleStore,
    iterations: usize,
    new_rows: usize,
) -> GeneralSccSummary {
    GeneralSccSummary {
        relations: scc
            .relations
            .iter()
            .map(|&relation| (relation, store.full[relation.0 as usize].len()))
            .collect(),
        iterations,
        new_rows,
    }
}

fn evaluate_clauses(
    clauses: &[RelationalClausePlan],
    store: &TupleStore,
) -> Result<Contributions, GeneralExecutionError> {
    let mut contributions = Contributions::new();
    for clause in clauses {
        if !clause.negative.is_empty() {
            return Err(GeneralExecutionError::NegationNotEnabled);
        }
        let tuples = evaluate_clause(clause, store);
        if let Some((_, existing)) = contributions
            .iter_mut()
            .find(|(target, _)| *target == clause.target)
        {
            existing.extend(tuples);
        } else {
            contributions.push((clause.target, tuples));
        }
    }
    Ok(contributions)
}

fn evaluate_clause(clause: &RelationalClausePlan, store: &TupleStore) -> TupleSet {
    let binding_count = clause
        .head
        .iter()
        .chain(clause.positive.iter().flat_map(|atom| atom.terms.iter()))
        .filter_map(|term| match term {
            PlanTerm::Binding(binding) => Some(binding.0 as usize + 1),
            PlanTerm::Value(_) => None,
        })
        .max()
        .unwrap_or(0);
    let mut bindings = vec![vec![None; binding_count]];
    for atom in &clause.positive {
        let rows = match atom.version {
            RelationVersion::Full => &store.full[atom.relation.0 as usize],
            RelationVersion::Delta => &store.delta[atom.relation.0 as usize],
            RelationVersion::Newt => unreachable!("NEWT is not a clause input"),
        };
        let mut joined = Vec::new();
        for binding in &bindings {
            for tuple in rows {
                if tuple.len() != atom.terms.len() {
                    continue;
                }
                let mut candidate = binding.clone();
                if match_atom(&mut candidate, &atom.terms, tuple) {
                    joined.push(candidate);
                }
            }
        }
        bindings = joined;
        if bindings.is_empty() {
            break;
        }
    }
    bindings
        .into_iter()
        .map(|binding| {
            clause
                .head
                .iter()
                .map(|term| match term {
                    PlanTerm::Binding(id) => binding[id.0 as usize]
                        .expect("validated head binding exists in positive body"),
                    PlanTerm::Value(value) => *value,
                })
                .collect()
        })
        .collect()
}

fn match_atom(binding: &mut [Option<u32>], terms: &[PlanTerm], tuple: &[u32]) -> bool {
    for (term, &value) in terms.iter().zip(tuple) {
        match term {
            PlanTerm::Value(expected) if *expected != value => return false,
            PlanTerm::Value(_) => {}
            PlanTerm::Binding(id) => match binding[id.0 as usize] {
                Some(expected) if expected != value => return false,
                Some(_) => {}
                None => binding[id.0 as usize] = Some(value),
            },
        }
    }
    true
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeneralExecutionError {
    NegationNotEnabled,
    IterationLimit { limit: usize },
}

impl std::fmt::Display for GeneralExecutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NegationNotEnabled => {
                formatter.write_str("stratified negation execution is not enabled")
            }
            Self::IterationLimit { limit } => {
                write!(
                    formatter,
                    "fixpoint did not converge within {limit} iterations"
                )
            }
        }
    }
}

impl std::error::Error for GeneralExecutionError {}

#[cfg(test)]
mod tests {
    use crate::{ProgramCatalog, parse_program, resolve_program};

    use super::*;

    fn execute(source: &str) -> (GeneralExecution, ProgramCatalog, GeneralProgramPlan) {
        let parsed = parse_program(source);
        assert_eq!(parsed.diagnostics, []);
        let mut catalog = ProgramCatalog::new();
        let resolved = resolve_program(&parsed.program, &mut catalog);
        assert_eq!(resolved.diagnostics, []);
        let plan = lower_general(&resolved.program).unwrap();
        let execution = execute_general(&plan, 32).unwrap();
        (execution, catalog, plan)
    }

    #[test]
    fn executes_general_nonrecursive_join_chains_and_constants() {
        let (execution, catalog, _) = execute(
            "
            parent('a, 'b). parent('b, 'c). tag('c, 'wanted).
            answer(x, z) :- parent(x, y), parent(y, z), tag(z, 'wanted).
            .output answer
            ",
        );
        let answer = relation_id(catalog.predicates.id("answer").unwrap());

        assert_eq!(execution.store.rows(answer).unwrap().len(), 1);
    }

    #[test]
    fn emits_and_executes_one_variant_per_recursive_atom() {
        let (execution, catalog, plan) = execute(
            "
            edge(1, 2). edge(2, 3). edge(3, 4).
            path(x, y) :- edge(x, y).
            path(x, z) :- path(x, y), path(y, z).
            .output path
            ",
        );
        let path = relation_id(catalog.predicates.id("path").unwrap());
        let recursive = plan
            .strata
            .iter()
            .flat_map(|stratum| &stratum.sccs)
            .find(|scc| scc.relations.contains(&path))
            .unwrap();

        assert_eq!(recursive.recursive_variants.len(), 2);
        assert_eq!(execution.store.rows(path).unwrap().len(), 6);
        assert!(execution.store.delta(path).unwrap().is_empty());
    }

    #[test]
    fn arbitrary_arity_relations_remain_distinct() {
        let (execution, catalog, _) =
            execute("triple('a, 'b, 'c). pair(x, z) :- triple(x, y, z). .output pair");
        let triple = relation_id(catalog.predicates.id("triple").unwrap());
        let pair = relation_id(catalog.predicates.id("pair").unwrap());

        assert_eq!(
            execution
                .store
                .rows(triple)
                .unwrap()
                .iter()
                .next()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(
            execution
                .store
                .rows(pair)
                .unwrap()
                .iter()
                .next()
                .unwrap()
                .len(),
            2
        );
    }
}
