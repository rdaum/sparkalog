//! Semi-naive fixpoint orchestration over relational plans.

pub use sparkalog_relational::RelationVersion;

/// Cardinality state at a semi-naive iteration boundary.
///
/// `next_delta_rows` is expected to have already been deduplicated both
/// internally and against `full_rows` by the relational plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixpointState {
    full_rows: usize,
    delta_rows: usize,
    iterations: usize,
}

impl FixpointState {
    pub fn seeded(seed_rows: usize) -> Self {
        Self {
            full_rows: seed_rows,
            delta_rows: seed_rows,
            iterations: 0,
        }
    }

    pub fn full_rows(self) -> usize {
        self.full_rows
    }

    pub fn delta_rows(self) -> usize {
        self.delta_rows
    }

    pub fn iterations(self) -> usize {
        self.iterations
    }

    pub fn reached_fixpoint(self) -> bool {
        self.delta_rows == 0
    }

    pub fn advance(&mut self, next_delta_rows: usize) {
        self.full_rows += next_delta_rows;
        self.delta_rows = next_delta_rows;
        self.iterations += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_delta_ends_the_fixpoint() {
        let mut state = FixpointState::seeded(4);
        state.advance(2);
        state.advance(0);

        assert!(state.reached_fixpoint());
        assert_eq!(state.full_rows(), 6);
        assert_eq!(state.iterations(), 2);
    }
}
