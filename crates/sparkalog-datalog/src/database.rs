use std::collections::HashSet;

use sparkalog_relational::{GeneralProgramPlan, RelationId};

use crate::{
    Diagnostic, GeneralExecution, GeneralExecutionError, GeneralSccSummary, InternedValue,
    LoweringError, PredicateId, ProgramCatalog, ResolvedProgram, execute_general, lower_general,
    parse_program, resolve_program,
};

pub struct Database {
    catalog: ProgramCatalog,
    program: Option<ResolvedProgram>,
    plan: Option<GeneralProgramPlan>,
    active_predicates: HashSet<PredicateId>,
    inserted_facts: Vec<(RelationId, Vec<u32>)>,
    execution: Option<GeneralExecution>,
}

impl Database {
    pub fn new() -> Self {
        Self {
            catalog: ProgramCatalog::new(),
            program: None,
            plan: None,
            active_predicates: HashSet::new(),
            inserted_facts: Vec::new(),
            execution: None,
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
        let active_predicates = resolved
            .program
            .rules
            .iter()
            .flat_map(|rule| {
                std::iter::once(rule.head.predicate)
                    .chain(rule.body.iter().map(|literal| literal.atom.predicate))
            })
            .collect();
        self.program = Some(resolved.program);
        self.plan = Some(plan);
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
        let mut plan = self.plan.clone().ok_or(DatabaseError::NoProgram)?;
        plan.facts.extend(self.inserted_facts.iter().cloned());
        let execution = execute_general(&plan, 10_000)?;
        let summary = RunSummary {
            sccs: execution.summaries.clone(),
        };
        self.execution = Some(execution);
        Ok(summary)
    }

    pub fn query(&self, predicate: &str) -> Result<QueryResult, DatabaseError> {
        let id = self
            .catalog
            .predicates
            .id(predicate)
            .filter(|id| self.active_predicates.contains(id))
            .ok_or_else(|| DatabaseError::UnknownPredicate(predicate.to_owned()))?;
        let execution = self.execution.as_ref().ok_or(DatabaseError::NotRun)?;
        let rows = execution
            .store
            .rows(RelationId(id.0))
            .expect("active relation exists")
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
    pub sccs: Vec<GeneralSccSummary>,
}

#[derive(Debug)]
pub enum DatabaseError {
    Parse(Vec<Diagnostic>),
    Resolve(Vec<Diagnostic>),
    Lowering(LoweringError),
    Execution(GeneralExecutionError),
    Catalog(crate::CatalogError),
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
            Self::Catalog(error) => error.fmt(formatter),
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

impl From<crate::CatalogError> for DatabaseError {
    fn from(error: crate::CatalogError) -> Self {
        Self::Catalog(error)
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
}
