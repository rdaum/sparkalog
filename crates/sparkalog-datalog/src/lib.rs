//! Datalog syntax, validation, stratification, and lowering.
//!
//! This crate is intentionally a frontend. It will lower safe, stratified
//! rules into plans from `sparkalog-relational` and fixpoint components from
//! `sparkalog-recursion`; it will not implement physical operators itself.

mod ast;
mod catalog;
mod database;
mod general;
mod ingest;
mod parser;
mod resolve;
mod schedule;
mod subset;

pub use ast::{
    ResolvedAtom, ResolvedLiteral, ResolvedProgram, ResolvedRule, ResolvedTerm, SourceAtom,
    SourceDeclaration, SourceLiteral, SourceProgram, SourceRule, SourceTerm, SourceValue, Span,
    Spanned, VariableId,
};
pub use catalog::{
    CatalogError, CatalogIoError, InternedValue, PredicateCatalog, PredicateId, PredicateMetadata,
    ProgramCatalog, ValueCatalog, ValueId,
};
pub use database::{Database, DatabaseError, ExecutionBackend, QueryResult, RunSummary};
pub use general::{
    GeneralExecution, GeneralExecutionError, GeneralSccSummary, PlanStatistics, TupleStore,
    execute_general, explain_general, lower_general, optimize_general,
};
pub use ingest::{DelimitedError, DelimitedOptions, parse_delimited_parallel};
pub use parser::{Diagnostic, ParseOutput, parse_program};
pub use resolve::{ResolveOutput, resolve_program};
pub use schedule::{
    ProgramSchedule, ScheduledScc, ScheduledStratum, StratificationError, dependencies, stratify,
};
pub use subset::{
    BinaryExecution, BinaryExecutionError, BinaryProgramPlan, BinarySccPlan, BinarySeedPlan,
    LoweringError, execute_binary, lower_binary,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependencyKind {
    Positive,
    Negative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Dependency {
    pub head: PredicateId,
    pub body: PredicateId,
    pub kind: DependencyKind,
}

/// A group of predicates whose completed relations may be consumed by later
/// strata. Negative dependencies must always point to an earlier stratum.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stratum {
    pub predicates: Vec<PredicateId>,
}
