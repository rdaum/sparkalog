use std::collections::HashSet;

use crate::{Dependency, DependencyKind, PredicateId, ResolvedProgram};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledScc {
    pub predicates: Vec<PredicateId>,
    pub recursive: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduledStratum {
    pub index: usize,
    pub sccs: Vec<ScheduledScc>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramSchedule {
    pub dependencies: Vec<Dependency>,
    pub strata: Vec<ScheduledStratum>,
}

pub fn dependencies(program: &ResolvedProgram) -> Vec<Dependency> {
    let mut result = Vec::new();
    for rule in &program.rules {
        for literal in &rule.body {
            let dependency = Dependency {
                head: rule.head.predicate,
                body: literal.atom.predicate,
                kind: if literal.negated {
                    DependencyKind::Negative
                } else {
                    DependencyKind::Positive
                },
            };
            if !result.contains(&dependency) {
                result.push(dependency);
            }
        }
    }
    result
}

pub fn stratify(program: &ResolvedProgram) -> Result<ProgramSchedule, StratificationError> {
    let dependencies = dependencies(program);
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
    let mut active = vec![false; predicate_count];
    for rule in &program.rules {
        active[rule.head.predicate.0 as usize] = true;
        for literal in &rule.body {
            active[literal.atom.predicate.0 as usize] = true;
        }
    }
    for &output in &program.outputs {
        active[output.0 as usize] = true;
    }
    let mut graph = vec![Vec::new(); predicate_count];
    let mut reverse = vec![Vec::new(); predicate_count];
    for dependency in &dependencies {
        let head = dependency.head.0 as usize;
        let body = dependency.body.0 as usize;
        graph[head].push(body);
        reverse[body].push(head);
    }
    let components = strongly_connected_components(&graph, &reverse, &active);
    let mut component_of = vec![usize::MAX; predicate_count];
    for (component, predicates) in components.iter().enumerate() {
        for &predicate in predicates {
            component_of[predicate.0 as usize] = component;
        }
    }
    for dependency in &dependencies {
        if dependency.kind == DependencyKind::Negative
            && component_of[dependency.head.0 as usize] == component_of[dependency.body.0 as usize]
        {
            return Err(StratificationError::NegativeCycle {
                head: dependency.head,
                body: dependency.body,
            });
        }
    }

    let mut component_strata = vec![0_usize; components.len()];
    for _ in 0..components.len() {
        let mut changed = false;
        for dependency in &dependencies {
            let head = component_of[dependency.head.0 as usize];
            let body = component_of[dependency.body.0 as usize];
            if head == body {
                continue;
            }
            let required =
                component_strata[body] + usize::from(dependency.kind == DependencyKind::Negative);
            if component_strata[head] < required {
                component_strata[head] = required;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let component_order = topological_components(&dependencies, &component_of, &components);
    let max_stratum = component_strata.iter().copied().max().unwrap_or(0);
    let mut strata = (0..=max_stratum)
        .map(|index| ScheduledStratum {
            index,
            sccs: Vec::new(),
        })
        .collect::<Vec<_>>();
    for component in component_order {
        let mut predicates = components[component].clone();
        predicates.sort_unstable();
        let recursive = predicates.len() > 1
            || dependencies.iter().any(|dependency| {
                dependency.kind == DependencyKind::Positive
                    && dependency.head == predicates[0]
                    && dependency.body == predicates[0]
            });
        strata[component_strata[component]].sccs.push(ScheduledScc {
            predicates,
            recursive,
        });
    }
    strata.retain(|stratum| !stratum.sccs.is_empty());
    Ok(ProgramSchedule {
        dependencies,
        strata,
    })
}

fn strongly_connected_components(
    graph: &[Vec<usize>],
    reverse: &[Vec<usize>],
    active: &[bool],
) -> Vec<Vec<PredicateId>> {
    fn visit(node: usize, graph: &[Vec<usize>], visited: &mut [bool], order: &mut Vec<usize>) {
        if visited[node] {
            return;
        }
        visited[node] = true;
        for &next in &graph[node] {
            visit(next, graph, visited, order);
        }
        order.push(node);
    }

    fn collect(
        node: usize,
        graph: &[Vec<usize>],
        visited: &mut [bool],
        result: &mut Vec<PredicateId>,
    ) {
        if visited[node] {
            return;
        }
        visited[node] = true;
        result.push(PredicateId(node as u32));
        for &next in &graph[node] {
            collect(next, graph, visited, result);
        }
    }

    let mut order = Vec::new();
    let mut visited = vec![false; graph.len()];
    for (node, &is_active) in active.iter().enumerate() {
        if is_active {
            visit(node, graph, &mut visited, &mut order);
        }
    }
    visited.fill(false);
    let mut components = Vec::new();
    for &node in order.iter().rev() {
        if !visited[node] {
            let mut component = Vec::new();
            collect(node, reverse, &mut visited, &mut component);
            components.push(component);
        }
    }
    components
}

fn topological_components(
    dependencies: &[Dependency],
    component_of: &[usize],
    components: &[Vec<PredicateId>],
) -> Vec<usize> {
    let mut outgoing = vec![HashSet::new(); components.len()];
    let mut indegree = vec![0_usize; components.len()];
    for dependency in dependencies {
        let consumer = component_of[dependency.head.0 as usize];
        let producer = component_of[dependency.body.0 as usize];
        if consumer != producer && outgoing[producer].insert(consumer) {
            indegree[consumer] += 1;
        }
    }
    let minimum_predicate = components
        .iter()
        .map(|component| component.iter().map(|id| id.0).min().unwrap_or(u32::MAX))
        .collect::<Vec<_>>();
    let mut available = indegree
        .iter()
        .enumerate()
        .filter_map(|(component, &degree)| (degree == 0).then_some(component))
        .collect::<Vec<_>>();
    let mut order = Vec::with_capacity(components.len());
    while !available.is_empty() {
        available
            .sort_unstable_by_key(|&component| std::cmp::Reverse(minimum_predicate[component]));
        let component = available.pop().expect("available is not empty");
        order.push(component);
        for &consumer in &outgoing[component] {
            indegree[consumer] -= 1;
            if indegree[consumer] == 0 {
                available.push(consumer);
            }
        }
    }
    order
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StratificationError {
    NegativeCycle {
        head: PredicateId,
        body: PredicateId,
    },
}

impl std::fmt::Display for StratificationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NegativeCycle { head, body } => write!(
                formatter,
                "negative dependency {} -> {} occurs inside a recursive SCC",
                head.0, body.0
            ),
        }
    }
}

impl std::error::Error for StratificationError {}

#[cfg(test)]
mod tests {
    use crate::{ProgramCatalog, parse_program, resolve_program};

    use super::*;

    fn schedule(source: &str) -> Result<(ProgramSchedule, ProgramCatalog), StratificationError> {
        let parsed = parse_program(source);
        assert_eq!(parsed.diagnostics, []);
        let mut catalog = ProgramCatalog::new();
        let resolved = resolve_program(&parsed.program, &mut catalog);
        assert_eq!(resolved.diagnostics, []);
        Ok((stratify(&resolved.program)?, catalog))
    }

    #[test]
    fn positive_mutual_recursion_forms_one_scc() {
        let (schedule, catalog) =
            schedule("a(x) :- seed(x). a(x) :- b(x). b(x) :- a(x). seed('x).").unwrap();
        let a = catalog.predicates.id("a").unwrap();
        let b = catalog.predicates.id("b").unwrap();

        let recursive = schedule.strata[0]
            .sccs
            .iter()
            .find(|scc| scc.predicates.contains(&a))
            .unwrap();
        assert_eq!(recursive.predicates, [a, b]);
        assert!(recursive.recursive);
    }

    #[test]
    fn negative_dependencies_advance_the_consumer_stratum() {
        let (schedule, catalog) =
            schedule("node('a). blocked('b). allowed(x) :- node(x), !blocked(x).").unwrap();
        let allowed = catalog.predicates.id("allowed").unwrap();

        assert_eq!(schedule.strata.len(), 2);
        assert!(schedule.strata[1].sccs[0].predicates.contains(&allowed));
    }

    #[test]
    fn negative_cycles_are_rejected() {
        let error = schedule("domain('x). a(x) :- domain(x), !b(x). b(x) :- domain(x), !a(x).")
            .unwrap_err();

        assert!(matches!(error, StratificationError::NegativeCycle { .. }));
    }

    #[test]
    fn producers_are_scheduled_before_consumers() {
        let (schedule, catalog) = schedule("a('x). b(x) :- a(x). c(x) :- b(x).").unwrap();
        let a = catalog.predicates.id("a").unwrap();
        let b = catalog.predicates.id("b").unwrap();
        let c = catalog.predicates.id("c").unwrap();
        let ordered = schedule.strata[0]
            .sccs
            .iter()
            .map(|scc| scc.predicates[0])
            .collect::<Vec<_>>();

        assert_eq!(ordered, [a, b, c]);
    }
}
