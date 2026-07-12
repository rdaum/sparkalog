use std::collections::HashSet;

use sparkalog_execution::CudaStream;
use sparkalog_recursion::IterationPolicies;
use sparkalog_relational::{GeneralProgramPlan, RelationId};

use crate::{
    BinaryExecution, BinaryExecutionError, BinaryProgramPlan, DelimitedError, DelimitedOptions,
    Diagnostic, GeneralExecution, GeneralExecutionError, GeneralSccSummary, InternedValue,
    LoweringError, PredicateId, ProgramCatalog, ResolvedProgram, execute_binary, execute_general,
    explain_general, lower_binary, lower_general, parse_delimited_parallel, parse_program,
    resolve_program,
};

enum DatabaseExecution {
    General(GeneralExecution),
    Binary(BinaryExecution),
}

pub struct Database {
    catalog: ProgramCatalog,
    program: Option<ResolvedProgram>,
    plan: Option<GeneralProgramPlan>,
    binary_plan: Option<BinaryProgramPlan>,
    active_predicates: HashSet<PredicateId>,
    inserted_facts: Vec<(RelationId, Vec<u32>)>,
    execution: Option<DatabaseExecution>,
}

impl Database {
    pub fn new() -> Self {
        Self {
            catalog: ProgramCatalog::new(),
            program: None,
            plan: None,
            binary_plan: None,
            active_predicates: HashSet::new(),
            inserted_facts: Vec::new(),
            execution: None,
        }
    }

    pub fn with_catalog(catalog: ProgramCatalog) -> Self {
        Self {
            catalog,
            ..Self::new()
        }
    }

    pub fn load_program(&mut self, source: &str) -> Result<(), DatabaseError> {
        let parsed = parse_program(source);
        if !parsed.diagnostics.is_empty() {
            return Err(DatabaseError::Parse(parsed.diagnostics));
        }
        let resolved = resolve_program(&parsed.program, &mut self.catalog);
        if !resolved.diagnostics.is_empty() {
            return Err(DatabaseError::Resolve(resolved.diagnostics));
        }
        let plan = lower_general(&resolved.program)?;
        let binary_plan = lower_binary(&resolved.program).ok();
        let active_predicates = resolved
            .program
            .rules
            .iter()
            .flat_map(|rule| {
                std::iter::once(rule.head.predicate)
                    .chain(rule.body.iter().map(|literal| literal.atom.predicate))
            })
            .chain(resolved.program.declarations.iter().copied())
            .collect();
        self.program = Some(resolved.program);
        self.plan = Some(plan);
        self.binary_plan = binary_plan;
        self.active_predicates = active_predicates;
        self.inserted_facts.clear();
        self.execution = None;
        Ok(())
    }

    pub fn insert<I, V>(&mut self, predicate: &str, values: I) -> Result<(), DatabaseError>
    where
        I: IntoIterator<Item = V>,
        V: Into<InternedValue>,
    {
        let id = self
            .catalog
            .predicates
            .id(predicate)
            .filter(|id| self.active_predicates.contains(id))
            .ok_or_else(|| DatabaseError::UnknownPredicate(predicate.to_owned()))?;
        let values = values.into_iter().map(Into::into).collect::<Vec<_>>();
        let expected = self
            .catalog
            .predicates
            .get(id)
            .and_then(|metadata| metadata.arity)
            .expect("active predicate has known arity");
        if values.len() != expected {
            return Err(DatabaseError::Arity {
                predicate: predicate.to_owned(),
                expected,
                actual: values.len(),
            });
        }
        let tuple = values
            .into_iter()
            .map(|value| self.catalog.values.intern(value).map(|id| id.0))
            .collect::<Result<Vec<_>, _>>()?;
        self.inserted_facts.push((RelationId(id.0), tuple));
        self.execution = None;
        Ok(())
    }

    pub fn run(&mut self) -> Result<RunSummary, DatabaseError> {
        if self.binary_plan.is_some()
            && let Ok(stream) = CudaStream::new()
        {
            return self.run_with_stream(Some(&stream));
        }
        self.run_general()
    }

    pub fn run_with_stream(
        &mut self,
        stream: Option<&CudaStream>,
    ) -> Result<RunSummary, DatabaseError> {
        if let Some(mut plan) = self.binary_plan.clone() {
            plan.facts
                .extend(self.inserted_facts.iter().filter_map(|(relation, tuple)| {
                    (tuple.len() == 2).then_some((*relation, [tuple[0], tuple[1]]))
                }));
            let execution = execute_binary(&plan, stream, IterationPolicies::default(), 10_000)?;
            let sccs = execution
                .summaries
                .iter()
                .map(|summary| GeneralSccSummary {
                    relations: summary.relation_rows.clone(),
                    iterations: summary.iterations,
                    new_rows: summary.total_new_rows,
                })
                .collect();
            self.execution = Some(DatabaseExecution::Binary(execution));
            return Ok(RunSummary {
                backend: ExecutionBackend::BinaryHybrid,
                sccs,
            });
        }
        self.run_general()
    }

    fn run_general(&mut self) -> Result<RunSummary, DatabaseError> {
        let mut plan = self.plan.clone().ok_or(DatabaseError::NoProgram)?;
        plan.facts.extend(self.inserted_facts.iter().cloned());
        let execution = execute_general(&plan, 10_000)?;
        let summary = RunSummary {
            backend: ExecutionBackend::GeneralCpu,
            sccs: execution.summaries.clone(),
        };
        self.execution = Some(DatabaseExecution::General(execution));
        Ok(summary)
    }

    pub fn load_delimited(
        &mut self,
        predicate: &str,
        source: &str,
        options: DelimitedOptions,
    ) -> Result<usize, DatabaseError> {
        let rows = parse_delimited_parallel(source, options)?;
        let count = rows.len();
        for row in rows {
            self.insert(predicate, row)?;
        }
        Ok(count)
    }

    pub fn query(&self, predicate: &str) -> Result<QueryResult, DatabaseError> {
        let id = self
            .catalog
            .predicates
            .id(predicate)
            .filter(|id| self.active_predicates.contains(id))
            .ok_or_else(|| DatabaseError::UnknownPredicate(predicate.to_owned()))?;
        let execution = self.execution.as_ref().ok_or(DatabaseError::NotRun)?;
        let encoded_rows = match execution {
            DatabaseExecution::General(execution) => execution
                .store
                .rows(RelationId(id.0))
                .expect("active relation exists")
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            DatabaseExecution::Binary(execution) => {
                let relation = execution.store.full(RelationId(id.0))?.view();
                (0..relation.len())
                    .map(|row| {
                        vec![
                            relation.column_slice(0).expect("binary relation")[row],
                            relation.column_slice(1).expect("binary relation")[row],
                        ]
                    })
                    .collect()
            }
        };
        let rows = encoded_rows
            .iter()
            .map(|tuple| {
                tuple
                    .iter()
                    .map(|&value| {
                        self.catalog
                            .values
                            .get(crate::ValueId(value))
                            .cloned()
                            .expect("stored value ID exists in catalog")
                    })
                    .collect()
            })
            .collect();
        Ok(QueryResult {
            predicate: predicate.to_owned(),
            rows,
        })
    }

    pub fn outputs(&self) -> Result<Vec<QueryResult>, DatabaseError> {
        let program = self.program.as_ref().ok_or(DatabaseError::NoProgram)?;
        program
            .outputs
            .iter()
            .map(|&id| {
                let name = &self
                    .catalog
                    .predicates
                    .get(id)
                    .expect("output predicate exists")
                    .name;
                self.query(name)
            })
            .collect()
    }

    pub fn catalog(&self) -> &ProgramCatalog {
        &self.catalog
    }

    pub fn explain(&self) -> Result<String, DatabaseError> {
        let plan = self.plan.as_ref().ok_or(DatabaseError::NoProgram)?;
        let mut explanation = format!(
            "preferred backend: {}\n",
            if self.binary_plan.is_some() {
                "binary hybrid CPU/CUDA"
            } else {
                "general native Rust"
            }
        );
        explanation.push_str(&explain_general(plan, &self.catalog));
        Ok(explanation)
    }
}

impl Default for Database {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryResult {
    pub predicate: String,
    pub rows: Vec<Vec<InternedValue>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunSummary {
    pub backend: ExecutionBackend,
    pub sccs: Vec<GeneralSccSummary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionBackend {
    GeneralCpu,
    BinaryHybrid,
}

#[derive(Debug)]
pub enum DatabaseError {
    Parse(Vec<Diagnostic>),
    Resolve(Vec<Diagnostic>),
    Lowering(LoweringError),
    Execution(GeneralExecutionError),
    BinaryExecution(BinaryExecutionError),
    Catalog(crate::CatalogError),
    Delimited(DelimitedError),
    NoProgram,
    NotRun,
    UnknownPredicate(String),
    Arity {
        predicate: String,
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for DatabaseError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(diagnostics) => write!(
                formatter,
                "Datalog parse failed: {}",
                diagnostics
                    .first()
                    .map_or("unknown error", |error| &error.message)
            ),
            Self::Resolve(diagnostics) => write!(
                formatter,
                "Datalog validation failed: {}",
                diagnostics
                    .first()
                    .map_or("unknown error", |error| &error.message)
            ),
            Self::Lowering(error) => error.fmt(formatter),
            Self::Execution(error) => error.fmt(formatter),
            Self::BinaryExecution(error) => error.fmt(formatter),
            Self::Catalog(error) => error.fmt(formatter),
            Self::Delimited(error) => error.fmt(formatter),
            Self::NoProgram => formatter.write_str("no Datalog program is loaded"),
            Self::NotRun => formatter.write_str("the Datalog program has not been run"),
            Self::UnknownPredicate(predicate) => {
                write!(
                    formatter,
                    "predicate {predicate} is not defined by this program"
                )
            }
            Self::Arity {
                predicate,
                expected,
                actual,
            } => write!(
                formatter,
                "predicate {predicate} expects {expected} values, not {actual}"
            ),
        }
    }
}

impl std::error::Error for DatabaseError {}

impl From<LoweringError> for DatabaseError {
    fn from(error: LoweringError) -> Self {
        Self::Lowering(error)
    }
}

impl From<GeneralExecutionError> for DatabaseError {
    fn from(error: GeneralExecutionError) -> Self {
        Self::Execution(error)
    }
}

impl From<BinaryExecutionError> for DatabaseError {
    fn from(error: BinaryExecutionError) -> Self {
        Self::BinaryExecution(error)
    }
}

impl From<sparkalog_recursion::RelationStoreError> for DatabaseError {
    fn from(error: sparkalog_recursion::RelationStoreError) -> Self {
        Self::BinaryExecution(BinaryExecutionError::Store(error))
    }
}

impl From<crate::CatalogError> for DatabaseError {
    fn from(error: crate::CatalogError) -> Self {
        Self::Catalog(error)
    }
}

impl From<DelimitedError> for DatabaseError {
    fn from(error: DelimitedError) -> Self {
        Self::Delimited(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn database_loads_runs_and_decodes_outputs() {
        let mut database = Database::new();
        database
            .load_program("edge('a, 'b). path(x, y) :- edge(x, y). .output path")
            .unwrap();

        database.run().unwrap();
        let output = database.outputs().unwrap();

        assert_eq!(output.len(), 1);
        assert_eq!(output[0].predicate, "path");
        assert_eq!(
            output[0].rows,
            [vec![
                InternedValue::Symbol("a".into()),
                InternedValue::Symbol("b".into())
            ]]
        );
    }

    #[test]
    fn inserted_facts_invalidate_and_extend_the_next_run() {
        let mut database = Database::new();
        database
            .load_program("edge('a, 'b). path(x, y) :- edge(x, y). .output path")
            .unwrap();
        database.run().unwrap();
        database.insert("edge", ["b", "c"]).unwrap();

        assert!(matches!(database.query("path"), Err(DatabaseError::NotRun)));
        database.run().unwrap();

        assert_eq!(database.query("path").unwrap().rows.len(), 2);
    }

    #[test]
    fn load_reports_frontend_diagnostics() {
        let mut database = Database::new();

        assert!(matches!(
            database.load_program("bad(x :- nope."),
            Err(DatabaseError::Parse(_))
        ));
        assert!(matches!(
            database.load_program("bad(x) :- !missing(x)."),
            Err(DatabaseError::Resolve(_))
        ));
    }

    #[test]
    fn database_loads_delimited_edb_rows() {
        let mut database = Database::new();
        database
            .load_program(
                ".decl edge(from:number, to:number) .input edge path(x, y) :- edge(x, y). .output path",
            )
            .unwrap();

        assert_eq!(
            database
                .load_delimited("edge", "1,2\n2,3\n", DelimitedOptions::default())
                .unwrap(),
            2
        );
        database.run().unwrap();

        assert_eq!(database.query("path").unwrap().rows.len(), 2);
    }

    #[test]
    fn database_selects_hybrid_binary_or_general_native_pipelines() {
        let mut binary = Database::new();
        binary
            .load_program(
                "edge(1, 2). edge(2, 3). path(x, y) :- edge(x, y). path(x, z) :- path(x, y), edge(y, z). .output path",
            )
            .unwrap();
        let mut general = Database::new();
        general
            .load_program("triple(1, 2, 3). pair(x, z) :- triple(x, y, z). .output pair")
            .unwrap();

        assert_eq!(
            binary.run().unwrap().backend,
            ExecutionBackend::BinaryHybrid
        );
        assert_eq!(general.run().unwrap().backend, ExecutionBackend::GeneralCpu);
        assert!(binary.explain().unwrap().contains("binary hybrid CPU/CUDA"));
        assert!(general.explain().unwrap().contains("general native Rust"));
    }
}
