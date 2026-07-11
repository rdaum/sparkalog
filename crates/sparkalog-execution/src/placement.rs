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
}
