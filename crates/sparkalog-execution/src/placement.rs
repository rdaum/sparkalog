/// The processor implementation selected for one physical operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    CpuSerial,
    CpuParallel,
    Gpu,
}

/// The processor which most recently populated the canonical input column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputProvenance {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterPlacementContext {
    pub rows: usize,
    pub input_provenance: InputProvenance,
    pub gpu_available: bool,
}

/// Cardinality thresholds for the filter/compaction operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterPlacementPolicy {
    pub cpu_produced_gpu_min_rows: usize,
    pub gpu_produced_gpu_min_rows: usize,
    pub gpu_unavailable_parallel_min_rows: usize,
}

impl FilterPlacementPolicy {
    /// Conservative thresholds from `benchmarks/filter-crossover.csv`, recorded
    /// on the local GB10 across 1%, 10%, 50%, and 90% selectivity.
    pub const MEASURED_GB10: Self = Self {
        cpu_produced_gpu_min_rows: 131_072,
        gpu_produced_gpu_min_rows: 32_768,
        gpu_unavailable_parallel_min_rows: 8_388_608,
    };

    pub fn place(self, context: FilterPlacementContext) -> Placement {
        let gpu_min_rows = match context.input_provenance {
            InputProvenance::Cpu => self.cpu_produced_gpu_min_rows,
            InputProvenance::Gpu => self.gpu_produced_gpu_min_rows,
        };
        if context.gpu_available && context.rows >= gpu_min_rows {
            Placement::Gpu
        } else if !context.gpu_available && context.rows >= self.gpu_unavailable_parallel_min_rows {
            Placement::CpuParallel
        } else {
            Placement::CpuSerial
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinPlacementContext {
    pub delta_rows: usize,
    pub gpu_available: bool,
}

/// Cardinality thresholds for indexed binary equality joins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JoinPlacementPolicy {
    pub gpu_min_delta_rows: usize,
    pub gpu_unavailable_parallel_min_rows: usize,
}

impl JoinPlacementPolicy {
    /// Thresholds from `benchmarks/join-crossover.csv`, recorded on the local
    /// GB10 for DBLP path expansion. These are workload-specific because join
    /// fanout and key skew affect the output work as well as delta cardinality.
    pub const MEASURED_GB10_DBLP: Self = Self {
        gpu_min_delta_rows: 2_048,
        gpu_unavailable_parallel_min_rows: 2_048,
    };

    pub fn place(self, context: JoinPlacementContext) -> Placement {
        if context.gpu_available && context.delta_rows >= self.gpu_min_delta_rows {
            Placement::Gpu
        } else if !context.gpu_available
            && context.delta_rows >= self.gpu_unavailable_parallel_min_rows
        {
            Placement::CpuParallel
        } else {
            Placement::CpuSerial
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistinctPlacementContext {
    pub rows: usize,
    pub input_provenance: InputProvenance,
    pub gpu_available: bool,
}

/// Cardinality thresholds for sorting and deduplicating binary tuples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistinctPlacementPolicy {
    pub cpu_produced_gpu_min_rows: usize,
    pub gpu_produced_gpu_min_rows: usize,
    pub gpu_unavailable_parallel_min_rows: usize,
}

impl DistinctPlacementPolicy {
    /// Conservative thresholds from `benchmarks/distinct-crossover.csv`,
    /// recorded on DBLP join candidates on the local GB10.
    pub const MEASURED_GB10_DBLP: Self = Self {
        cpu_produced_gpu_min_rows: 32_768,
        gpu_produced_gpu_min_rows: 32_768,
        gpu_unavailable_parallel_min_rows: 131_072,
    };

    pub fn place(self, context: DistinctPlacementContext) -> Placement {
        let gpu_min_rows = match context.input_provenance {
            InputProvenance::Cpu => self.cpu_produced_gpu_min_rows,
            InputProvenance::Gpu => self.gpu_produced_gpu_min_rows,
        };
        if context.gpu_available && context.rows >= gpu_min_rows {
            Placement::Gpu
        } else if !context.gpu_available && context.rows >= self.gpu_unavailable_parallel_min_rows {
            Placement::CpuParallel
        } else {
            Placement::CpuSerial
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AntiJoinPlacementContext {
    pub left_rows: usize,
    pub input_provenance: InputProvenance,
    pub gpu_available: bool,
}

/// Cardinality thresholds for sorted binary anti-join.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AntiJoinPlacementPolicy {
    pub cpu_produced_gpu_min_rows: usize,
    pub gpu_produced_gpu_min_rows: usize,
    pub cpu_produced_parallel_min_rows: usize,
    pub gpu_produced_parallel_min_rows: usize,
}

impl AntiJoinPlacementPolicy {
    /// Conservative thresholds from `benchmarks/anti-join-crossover.csv`,
    /// recorded for DBLP candidate-minus-full relations on the local GB10.
    pub const MEASURED_GB10_DBLP: Self = Self {
        cpu_produced_gpu_min_rows: 32_768,
        gpu_produced_gpu_min_rows: 16_384,
        cpu_produced_parallel_min_rows: 32_768,
        gpu_produced_parallel_min_rows: 262_144,
    };

    pub fn place(self, context: AntiJoinPlacementContext) -> Placement {
        let (gpu_min_rows, parallel_min_rows) = match context.input_provenance {
            InputProvenance::Cpu => (
                self.cpu_produced_gpu_min_rows,
                self.cpu_produced_parallel_min_rows,
            ),
            InputProvenance::Gpu => (
                self.gpu_produced_gpu_min_rows,
                self.gpu_produced_parallel_min_rows,
            ),
        };
        if context.gpu_available && context.left_rows >= gpu_min_rows {
            Placement::Gpu
        } else if !context.gpu_available && context.left_rows >= parallel_min_rows {
            Placement::CpuParallel
        } else {
            Placement::CpuSerial
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnionPlacementContext {
    pub left_rows: usize,
    pub right_rows: usize,
    pub input_provenance: InputProvenance,
    pub gpu_available: bool,
}

/// Total-cardinality thresholds for sorted binary union.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnionPlacementPolicy {
    pub cpu_produced_gpu_min_rows: usize,
    pub gpu_produced_gpu_min_rows: usize,
    pub gpu_unavailable_parallel_min_rows: usize,
}

impl UnionPlacementPolicy {
    /// Conservative bounds from `benchmarks/union-crossover.csv`, recorded for
    /// DBLP `FULL union NEWT` updates on the local GB10.
    pub const MEASURED_GB10_DBLP: Self = Self {
        cpu_produced_gpu_min_rows: 1_048_576,
        gpu_produced_gpu_min_rows: 1_048_576,
        gpu_unavailable_parallel_min_rows: 4_194_304,
    };

    pub fn place(self, context: UnionPlacementContext) -> Placement {
        let rows = context.left_rows.saturating_add(context.right_rows);
        let gpu_min_rows = match context.input_provenance {
            InputProvenance::Cpu => self.cpu_produced_gpu_min_rows,
            InputProvenance::Gpu => self.gpu_produced_gpu_min_rows,
        };
        if context.gpu_available && rows >= gpu_min_rows {
            Placement::Gpu
        } else if !context.gpu_available && rows >= self.gpu_unavailable_parallel_min_rows {
            Placement::CpuParallel
        } else {
            Placement::CpuSerial
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_policy_accounts_for_input_provenance() {
        let policy = FilterPlacementPolicy::MEASURED_GB10;
        assert_eq!(
            policy.place(FilterPlacementContext {
                rows: 32_768,
                input_provenance: InputProvenance::Cpu,
                gpu_available: true,
            }),
            Placement::CpuSerial
        );
        assert_eq!(
            policy.place(FilterPlacementContext {
                rows: 32_768,
                input_provenance: InputProvenance::Gpu,
                gpu_available: true,
            }),
            Placement::Gpu
        );
        assert_eq!(
            policy.place(FilterPlacementContext {
                rows: 131_072,
                input_provenance: InputProvenance::Cpu,
                gpu_available: true,
            }),
            Placement::Gpu
        );
    }

    #[test]
    fn parallel_rust_is_only_the_measured_gpu_unavailable_fallback() {
        let policy = FilterPlacementPolicy::MEASURED_GB10;
        assert_eq!(
            policy.place(FilterPlacementContext {
                rows: 8_388_607,
                input_provenance: InputProvenance::Cpu,
                gpu_available: false,
            }),
            Placement::CpuSerial
        );
        assert_eq!(
            policy.place(FilterPlacementContext {
                rows: 8_388_608,
                input_provenance: InputProvenance::Cpu,
                gpu_available: false,
            }),
            Placement::CpuParallel
        );
        assert_eq!(
            policy.place(FilterPlacementContext {
                rows: 8_388_608,
                input_provenance: InputProvenance::Cpu,
                gpu_available: true,
            }),
            Placement::Gpu
        );
    }

    #[test]
    fn measured_join_policy_uses_the_dblp_crossover() {
        let policy = JoinPlacementPolicy::MEASURED_GB10_DBLP;
        assert_eq!(
            policy.place(JoinPlacementContext {
                delta_rows: 512,
                gpu_available: true,
            }),
            Placement::CpuSerial
        );
        assert_eq!(
            policy.place(JoinPlacementContext {
                delta_rows: 2_048,
                gpu_available: true,
            }),
            Placement::Gpu
        );
        assert_eq!(
            policy.place(JoinPlacementContext {
                delta_rows: 2_048,
                gpu_available: false,
            }),
            Placement::CpuParallel
        );
    }

    #[test]
    fn measured_distinct_policy_uses_candidate_cardinality() {
        let policy = DistinctPlacementPolicy::MEASURED_GB10_DBLP;
        for input_provenance in [InputProvenance::Cpu, InputProvenance::Gpu] {
            assert_eq!(
                policy.place(DistinctPlacementContext {
                    rows: 7_424,
                    input_provenance,
                    gpu_available: true,
                }),
                Placement::CpuSerial
            );
            assert_eq!(
                policy.place(DistinctPlacementContext {
                    rows: 32_768,
                    input_provenance,
                    gpu_available: true,
                }),
                Placement::Gpu
            );
            assert_eq!(
                policy.place(DistinctPlacementContext {
                    rows: 131_072,
                    input_provenance,
                    gpu_available: false,
                }),
                Placement::CpuParallel
            );
        }
    }

    #[test]
    fn measured_anti_join_policy_accounts_for_provenance() {
        let policy = AntiJoinPlacementPolicy::MEASURED_GB10_DBLP;
        assert_eq!(
            policy.place(AntiJoinPlacementContext {
                left_rows: 16_384,
                input_provenance: InputProvenance::Cpu,
                gpu_available: true,
            }),
            Placement::CpuSerial
        );
        assert_eq!(
            policy.place(AntiJoinPlacementContext {
                left_rows: 16_384,
                input_provenance: InputProvenance::Gpu,
                gpu_available: true,
            }),
            Placement::Gpu
        );
        assert_eq!(
            policy.place(AntiJoinPlacementContext {
                left_rows: 32_768,
                input_provenance: InputProvenance::Cpu,
                gpu_available: false,
            }),
            Placement::CpuParallel
        );
        assert_eq!(
            policy.place(AntiJoinPlacementContext {
                left_rows: 131_072,
                input_provenance: InputProvenance::Gpu,
                gpu_available: false,
            }),
            Placement::CpuSerial
        );
        assert_eq!(
            policy.place(AntiJoinPlacementContext {
                left_rows: 262_144,
                input_provenance: InputProvenance::Gpu,
                gpu_available: false,
            }),
            Placement::CpuParallel
        );
    }

    #[test]
    fn measured_union_policy_uses_total_merge_cardinality() {
        let policy = UnionPlacementPolicy::MEASURED_GB10_DBLP;
        assert_eq!(
            policy.place(UnionPlacementContext {
                left_rows: 1_000_000,
                right_rows: 40_000,
                input_provenance: InputProvenance::Cpu,
                gpu_available: true,
            }),
            Placement::CpuSerial
        );
        assert_eq!(
            policy.place(UnionPlacementContext {
                left_rows: 1_000_000,
                right_rows: 50_000,
                input_provenance: InputProvenance::Gpu,
                gpu_available: true,
            }),
            Placement::Gpu
        );
        assert_eq!(
            policy.place(UnionPlacementContext {
                left_rows: 2_000_000,
                right_rows: 2_194_304,
                input_provenance: InputProvenance::Cpu,
                gpu_available: false,
            }),
            Placement::CpuParallel
        );
    }
}
