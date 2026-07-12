use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PredicateId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InternedValue {
    U32(u32),
    String(String),
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateMetadata {
    pub name: String,
    pub arity: Option<usize>,
}

#[derive(Debug, Default)]
pub struct PredicateCatalog {
    by_name: HashMap<String, PredicateId>,
    predicates: Vec<PredicateMetadata>,
}

impl PredicateCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, name: &str) -> Result<PredicateId, CatalogError> {
        if let Some(&id) = self.by_name.get(name) {
            return Ok(id);
        }
        let id = PredicateId(next_id(self.predicates.len(), "predicate")?);
        let owned = name.to_owned();
        self.predicates.push(PredicateMetadata {
            name: owned.clone(),
            arity: None,
        });
        self.by_name.insert(owned, id);
        Ok(id)
    }

    pub fn declare(&mut self, name: &str, arity: usize) -> Result<PredicateId, CatalogError> {
        let id = self.intern(name)?;
        let metadata = &mut self.predicates[id.0 as usize];
        match metadata.arity {
            Some(existing) if existing != arity => Err(CatalogError::ArityConflict {
                predicate: name.to_owned(),
                existing,
                declared: arity,
            }),
            _ => {
                metadata.arity = Some(arity);
                Ok(id)
            }
        }
    }

    pub fn id(&self, name: &str) -> Option<PredicateId> {
        self.by_name.get(name).copied()
    }

    pub fn get(&self, id: PredicateId) -> Option<&PredicateMetadata> {
        self.predicates.get(id.0 as usize)
    }

    pub fn len(&self) -> usize {
        self.predicates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.predicates.is_empty()
    }
}

#[derive(Debug, Default)]
pub struct ValueCatalog {
    by_value: HashMap<InternedValue, ValueId>,
    values: Vec<InternedValue>,
}

impl ValueCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, value: InternedValue) -> Result<ValueId, CatalogError> {
        if let Some(&id) = self.by_value.get(&value) {
            return Ok(id);
        }
        let id = ValueId(next_id(self.values.len(), "value")?);
        self.values.push(value.clone());
        self.by_value.insert(value, id);
        Ok(id)
    }

    pub fn id(&self, value: &InternedValue) -> Option<ValueId> {
        self.by_value.get(value).copied()
    }

    pub fn get(&self, id: ValueId) -> Option<&InternedValue> {
        self.values.get(id.0 as usize)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[derive(Debug, Default)]
pub struct ProgramCatalog {
    pub predicates: PredicateCatalog,
    pub values: ValueCatalog,
}

impl ProgramCatalog {
    pub fn new() -> Self {
        Self::default()
    }
}

fn next_id(length: usize, kind: &'static str) -> Result<u32, CatalogError> {
    u32::try_from(length).map_err(|_| CatalogError::IdSpaceExhausted { kind })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogError {
    IdSpaceExhausted {
        kind: &'static str,
    },
    ArityConflict {
        predicate: String,
        existing: usize,
        declared: usize,
    },
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IdSpaceExhausted { kind } => write!(formatter, "{kind} ID space is exhausted"),
            Self::ArityConflict {
                predicate,
                existing,
                declared,
            } => write!(
                formatter,
                "predicate {predicate} already has arity {existing}, not {declared}"
            ),
        }
    }
}

impl std::error::Error for CatalogError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicate_ids_are_stable_and_enforce_arity() {
        let mut catalog = PredicateCatalog::new();
        let edge = catalog.declare("edge", 2).unwrap();

        assert_eq!(catalog.declare("edge", 2).unwrap(), edge);
        assert_eq!(catalog.id("edge"), Some(edge));
        assert_eq!(catalog.get(edge).unwrap().name, "edge");
        assert_eq!(catalog.get(edge).unwrap().arity, Some(2));
        assert!(matches!(
            catalog.declare("edge", 3),
            Err(CatalogError::ArityConflict { .. })
        ));
    }

    #[test]
    fn value_kinds_have_distinct_stable_ids() {
        let mut catalog = ValueCatalog::new();
        let number = catalog.intern(InternedValue::U32(42)).unwrap();
        let string = catalog
            .intern(InternedValue::String("42".to_owned()))
            .unwrap();
        let symbol = catalog
            .intern(InternedValue::Symbol("42".to_owned()))
            .unwrap();

        assert_ne!(number, string);
        assert_ne!(string, symbol);
        assert_eq!(catalog.intern(InternedValue::U32(42)).unwrap(), number);
        assert_eq!(
            catalog.get(string),
            Some(&InternedValue::String("42".into()))
        );
    }
}
