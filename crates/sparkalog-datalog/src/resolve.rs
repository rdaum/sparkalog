use std::collections::{HashMap, HashSet};

use crate::{
    CatalogError, Diagnostic, InternedValue, ProgramCatalog, ResolvedAtom, ResolvedLiteral,
    ResolvedProgram, ResolvedRule, ResolvedTerm, SourceAtom, SourceProgram, SourceRule, SourceTerm,
    SourceValue, Span, VariableId,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolveOutput {
    pub program: ResolvedProgram,
    pub diagnostics: Vec<Diagnostic>,
}

impl ResolveOutput {
    pub fn is_success(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

pub fn resolve_program(source: &SourceProgram, catalog: &mut ProgramCatalog) -> ResolveOutput {
    let mut diagnostics = Vec::new();
    let mut rules = Vec::with_capacity(source.rules.len());
    for rule in &source.rules {
        match resolve_rule(rule, catalog) {
            Ok(rule) => rules.push(rule),
            Err(mut errors) => diagnostics.append(&mut errors),
        }
    }
    let mut outputs = Vec::with_capacity(source.outputs.len());
    for output in &source.outputs {
        match catalog.predicates.id(&output.value) {
            Some(id)
                if catalog
                    .predicates
                    .get(id)
                    .is_some_and(|metadata| metadata.arity.is_some()) =>
            {
                outputs.push(id);
            }
            _ => diagnostics.push(Diagnostic {
                message: format!("output predicate {} is not defined", output.value),
                span: output.span,
            }),
        }
    }
    ResolveOutput {
        program: ResolvedProgram { rules, outputs },
        diagnostics,
    }
}

fn resolve_rule(
    source: &SourceRule,
    catalog: &mut ProgramCatalog,
) -> Result<ResolvedRule, Vec<Diagnostic>> {
    let mut context = RuleContext::default();
    let mut diagnostics = Vec::new();
    let head = resolve_atom(&source.head, catalog, &mut context, &mut diagnostics);
    let mut body = Vec::with_capacity(source.body.len());
    for literal in &source.body {
        let atom = resolve_atom(&literal.atom, catalog, &mut context, &mut diagnostics);
        if let Some(atom) = atom {
            body.push(ResolvedLiteral {
                negated: literal.negated,
                atom,
                span: literal.span,
            });
        }
    }

    let positive_variables = body
        .iter()
        .filter(|literal| !literal.negated)
        .flat_map(|literal| variables(&literal.atom))
        .collect::<HashSet<_>>();
    if let Some(head) = &head {
        for (term, source_term) in head.terms.iter().zip(&source.head.terms) {
            if let ResolvedTerm::Variable(variable) = term
                && !positive_variables.contains(variable)
            {
                diagnostics.push(Diagnostic {
                    message: format!(
                        "head variable {} is not bound by a positive body atom",
                        context.variable_name(*variable)
                    ),
                    span: source_term.span,
                });
            }
        }
    }
    for (resolved, source_literal) in body.iter().zip(&source.body) {
        if !resolved.negated {
            continue;
        }
        for (term, source_term) in resolved.atom.terms.iter().zip(&source_literal.atom.terms) {
            if let ResolvedTerm::Variable(variable) = term
                && !positive_variables.contains(variable)
            {
                diagnostics.push(Diagnostic {
                    message: format!(
                        "variable {} in a negated atom is not bound by a positive body atom",
                        context.variable_name(*variable)
                    ),
                    span: source_term.span,
                });
            }
        }
    }

    if diagnostics.is_empty() {
        Ok(ResolvedRule {
            head: head.expect("successful resolution produced a head"),
            body,
            variable_names: context.variable_names,
            span: source.span,
        })
    } else {
        Err(diagnostics)
    }
}

fn resolve_atom(
    source: &SourceAtom,
    catalog: &mut ProgramCatalog,
    context: &mut RuleContext,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<ResolvedAtom> {
    let predicate = match catalog
        .predicates
        .declare(&source.predicate.value, source.terms.len())
    {
        Ok(id) => Some(id),
        Err(error) => {
            diagnostics.push(catalog_diagnostic(error, source.predicate.span));
            None
        }
    };
    let mut terms = Vec::with_capacity(source.terms.len());
    for term in &source.terms {
        match resolve_term(&term.value, catalog, context) {
            Ok(resolved) => terms.push(resolved),
            Err(error) => diagnostics.push(catalog_diagnostic(error, term.span)),
        }
    }
    predicate.map(|predicate| ResolvedAtom {
        predicate,
        terms,
        span: source.span,
    })
}

fn resolve_term(
    source: &SourceTerm,
    catalog: &mut ProgramCatalog,
    context: &mut RuleContext,
) -> Result<ResolvedTerm, CatalogError> {
    match source {
        SourceTerm::Variable(name) => Ok(ResolvedTerm::Variable(context.variable(name))),
        SourceTerm::Constant(value) => Ok(ResolvedTerm::Value(
            catalog.values.intern(interned_value(value))?,
        )),
    }
}

fn interned_value(value: &SourceValue) -> InternedValue {
    match value {
        SourceValue::U32(value) => InternedValue::U32(*value),
        SourceValue::String(value) => InternedValue::String(value.clone()),
        SourceValue::Symbol(value) => InternedValue::Symbol(value.clone()),
    }
}

fn variables(atom: &ResolvedAtom) -> impl Iterator<Item = VariableId> + '_ {
    atom.terms.iter().filter_map(|term| match term {
        ResolvedTerm::Variable(variable) => Some(*variable),
        ResolvedTerm::Value(_) => None,
    })
}

fn catalog_diagnostic(error: CatalogError, span: Span) -> Diagnostic {
    Diagnostic {
        message: error.to_string(),
        span,
    }
}

#[derive(Default)]
struct RuleContext {
    variables: HashMap<String, VariableId>,
    variable_names: Vec<String>,
}

impl RuleContext {
    fn variable(&mut self, name: &str) -> VariableId {
        if name != "_"
            && let Some(&id) = self.variables.get(name)
        {
            return id;
        }
        let id = VariableId(
            u32::try_from(self.variable_names.len()).expect("rule has too many variables"),
        );
        self.variable_names.push(name.to_owned());
        if name != "_" {
            self.variables.insert(name.to_owned(), id);
        }
        id
    }

    fn variable_name(&self, id: VariableId) -> &str {
        &self.variable_names[id.0 as usize]
    }
}

#[cfg(test)]
mod tests {
    use crate::{InternedValue, parse_program};

    use super::*;

    fn resolved(source: &str) -> (ResolveOutput, ProgramCatalog) {
        let parsed = parse_program(source);
        assert_eq!(parsed.diagnostics, []);
        let mut catalog = ProgramCatalog::new();
        let output = resolve_program(&parsed.program, &mut catalog);
        (output, catalog)
    }

    #[test]
    fn resolves_predicates_values_and_rule_local_variables() {
        let (output, catalog) = resolved("edge('a, 'b). path(x, y) :- edge(x, y). .output path");

        assert_eq!(output.diagnostics, []);
        assert_eq!(output.program.rules.len(), 2);
        assert_eq!(output.program.rules[1].variable_names, ["x", "y"]);
        assert_eq!(catalog.predicates.len(), 2);
        assert_eq!(catalog.values.len(), 2);
        assert!(
            catalog
                .values
                .id(&InternedValue::Symbol("a".into()))
                .is_some()
        );
    }

    #[test]
    fn rejects_inconsistent_arity() {
        let (output, _) = resolved("edge('a, 'b). edge('a).");

        assert_eq!(output.program.rules.len(), 1);
        assert!(
            output.diagnostics[0]
                .message
                .contains("already has arity 2")
        );
    }

    #[test]
    fn rejects_unsafe_head_and_negated_variables() {
        let (output, _) = resolved("bad(x) :- edge(y, z), !blocked(w).");

        assert!(output.program.rules.is_empty());
        assert_eq!(output.diagnostics.len(), 2);
        assert!(output.diagnostics[0].message.contains("head variable x"));
        assert!(output.diagnostics[1].message.contains("negated atom"));
    }

    #[test]
    fn repeated_wildcards_are_distinct_variables() {
        let (output, _) = resolved("seen(x) :- edge(x, _), edge(_, x).");

        assert_eq!(output.diagnostics, []);
        assert_eq!(output.program.rules[0].variable_names, ["x", "_", "_"]);
    }

    #[test]
    fn output_must_name_a_defined_predicate() {
        let (output, _) = resolved("edge(1, 2). .output missing");

        assert_eq!(output.program.outputs, []);
        assert!(output.diagnostics[0].message.contains("is not defined"));
    }
}
