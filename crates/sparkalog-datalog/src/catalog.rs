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

impl From<u32> for InternedValue {
    fn from(value: u32) -> Self {
        Self::U32(value)
    }
}

impl From<String> for InternedValue {
    fn from(value: String) -> Self {
        Self::Symbol(value)
    }
}

impl From<&str> for InternedValue {
    fn from(value: &str) -> Self {
        Self::Symbol(value.to_owned())
    }
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

    pub fn write_to(&self, mut writer: impl std::io::Write) -> Result<(), CatalogIoError> {
        writer.write_all(b"SPKCAT01")?;
        write_u32(&mut writer, self.predicates.predicates.len())?;
        for predicate in &self.predicates.predicates {
            write_bytes(&mut writer, predicate.name.as_bytes())?;
            let arity = match predicate.arity {
                None => u32::MAX,
                Some(arity) => {
                    let arity = u32::try_from(arity)
                        .map_err(|_| CatalogIoError::Invalid("predicate arity is too large"))?;
                    if arity == u32::MAX {
                        return Err(CatalogIoError::Invalid("predicate arity is too large"));
                    }
                    arity
                }
            };
            write_u32(&mut writer, arity as usize)?;
        }
        write_u32(&mut writer, self.values.values.len())?;
        for value in &self.values.values {
            match value {
                InternedValue::U32(value) => {
                    writer.write_all(&[0])?;
                    writer.write_all(&value.to_le_bytes())?;
                }
                InternedValue::String(value) => {
                    writer.write_all(&[1])?;
                    write_bytes(&mut writer, value.as_bytes())?;
                }
                InternedValue::Symbol(value) => {
                    writer.write_all(&[2])?;
                    write_bytes(&mut writer, value.as_bytes())?;
                }
            }
        }
        Ok(())
    }

    pub fn read_from(mut reader: impl std::io::Read) -> Result<Self, CatalogIoError> {
        let mut magic = [0_u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"SPKCAT01" {
            return Err(CatalogIoError::Invalid("invalid catalog header"));
        }
        let mut catalog = Self::new();
        let predicates = read_u32(&mut reader)?;
        for _ in 0..predicates {
            let name = read_string(&mut reader)?;
            let arity = read_u32(&mut reader)?;
            if arity == u32::MAX {
                catalog.predicates.intern(&name)?;
            } else {
                catalog.predicates.declare(&name, arity as usize)?;
            }
        }
        let values = read_u32(&mut reader)?;
        for _ in 0..values {
            let mut tag = [0_u8; 1];
            reader.read_exact(&mut tag)?;
            let value = match tag[0] {
                0 => InternedValue::U32(read_u32(&mut reader)?),
                1 => InternedValue::String(read_string(&mut reader)?),
                2 => InternedValue::Symbol(read_string(&mut reader)?),
                _ => return Err(CatalogIoError::Invalid("unknown catalog value tag")),
            };
            catalog.values.intern(value)?;
        }
        Ok(catalog)
    }
}

fn write_u32(writer: &mut impl std::io::Write, value: usize) -> Result<(), CatalogIoError> {
    let value =
        u32::try_from(value).map_err(|_| CatalogIoError::Invalid("catalog is too large"))?;
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_bytes(writer: &mut impl std::io::Write, bytes: &[u8]) -> Result<(), CatalogIoError> {
    write_u32(writer, bytes.len())?;
    writer.write_all(bytes)?;
    Ok(())
}

fn read_u32(reader: &mut impl std::io::Read) -> Result<u32, CatalogIoError> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_string(reader: &mut impl std::io::Read) -> Result<String, CatalogIoError> {
    let length = read_u32(reader)? as usize;
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|_| CatalogIoError::Invalid("catalog text is not UTF-8"))
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

#[derive(Debug)]
pub enum CatalogIoError {
    Io(std::io::Error),
    Catalog(CatalogError),
    Invalid(&'static str),
}

impl std::fmt::Display for CatalogIoError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Catalog(error) => error.fmt(formatter),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for CatalogIoError {}

impl From<std::io::Error> for CatalogIoError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<CatalogError> for CatalogIoError {
    fn from(error: CatalogError) -> Self {
        Self::Catalog(error)
    }
}

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

    #[test]
    fn catalog_round_trip_preserves_every_id() {
        let mut catalog = ProgramCatalog::new();
        let edge = catalog.predicates.declare("edge", 2).unwrap();
        let number = catalog.values.intern(InternedValue::U32(42)).unwrap();
        let symbol = catalog
            .values
            .intern(InternedValue::Symbol("answer".into()))
            .unwrap();
        let mut bytes = Vec::new();
        catalog.write_to(&mut bytes).unwrap();

        let restored = ProgramCatalog::read_from(bytes.as_slice()).unwrap();

        assert_eq!(restored.predicates.id("edge"), Some(edge));
        assert_eq!(restored.values.get(number), Some(&InternedValue::U32(42)));
        assert_eq!(
            restored.values.get(symbol),
            Some(&InternedValue::Symbol("answer".into()))
        );
    }
}
