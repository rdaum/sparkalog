use crate::{PredicateId, ValueId};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn join(self, other: Self) -> Self {
        Self {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spanned<T> {
    pub value: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(value: T, span: Span) -> Self {
        Self { value, span }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SourceValue {
    U32(u32),
    String(String),
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceTerm {
    Variable(String),
    Constant(SourceValue),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceAtom {
    pub predicate: Spanned<String>,
    pub terms: Vec<Spanned<SourceTerm>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLiteral {
    pub negated: bool,
    pub atom: SourceAtom,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceRule {
    pub head: SourceAtom,
    pub body: Vec<SourceLiteral>,
    pub span: Span,
}

impl SourceRule {
    pub fn is_fact(&self) -> bool {
        self.body.is_empty()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceProgram {
    pub rules: Vec<SourceRule>,
    pub outputs: Vec<Spanned<String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VariableId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResolvedTerm {
    Variable(VariableId),
    Value(ValueId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAtom {
    pub predicate: PredicateId,
    pub terms: Vec<ResolvedTerm>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedLiteral {
    pub negated: bool,
    pub atom: ResolvedAtom,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRule {
    pub head: ResolvedAtom,
    pub body: Vec<ResolvedLiteral>,
    pub variable_names: Vec<String>,
    pub span: Span,
}

impl ResolvedRule {
    pub fn is_fact(&self) -> bool {
        self.body.is_empty()
    }

    pub fn variable_name(&self, id: VariableId) -> Option<&str> {
        self.variable_names.get(id.0 as usize).map(String::as_str)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolvedProgram {
    pub rules: Vec<ResolvedRule>,
    pub outputs: Vec<PredicateId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facts_are_rules_without_body_literals() {
        let atom = SourceAtom {
            predicate: Spanned::new("edge".into(), Span::new(0, 4)),
            terms: vec![],
            span: Span::new(0, 6),
        };
        let fact = SourceRule {
            head: atom,
            body: vec![],
            span: Span::new(0, 7),
        };

        assert!(fact.is_fact());
    }

    #[test]
    fn resolved_variable_names_are_rule_local() {
        let rule = ResolvedRule {
            head: ResolvedAtom {
                predicate: PredicateId(0),
                terms: vec![ResolvedTerm::Variable(VariableId(0))],
                span: Span::default(),
            },
            body: vec![],
            variable_names: vec!["x".into()],
            span: Span::default(),
        };

        assert_eq!(rule.variable_name(VariableId(0)), Some("x"));
        assert_eq!(rule.variable_name(VariableId(1)), None);
    }
}
