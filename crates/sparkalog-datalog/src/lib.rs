//! Datalog syntax, validation, stratification, and lowering.
//!
//! This crate is intentionally a frontend. It will lower safe, stratified
//! rules into plans from `sparkalog-relational` and fixpoint components from
//! `sparkalog-recursion`; it will not implement physical operators itself.

mod catalog;

pub use catalog::{
    CatalogError, InternedValue, PredicateCatalog, PredicateId, PredicateMetadata, ProgramCatalog,
    ValueCatalog, ValueId,
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
