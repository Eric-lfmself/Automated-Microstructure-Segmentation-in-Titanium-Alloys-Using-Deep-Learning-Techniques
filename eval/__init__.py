"""Global segmentation metrics and bounded CPU timing (paper Sections 3.2–3.4)."""

from .metrics import PAPER_REFERENCE, SegmentationMetrics, paper_reference_audit, render_comparison_table, render_per_class_table
from .timing import benchmark_model_cpu, time_pipeline_cpu

__all__ = ["PAPER_REFERENCE", "SegmentationMetrics", "paper_reference_audit", "render_comparison_table",
           "render_per_class_table", "benchmark_model_cpu", "time_pipeline_cpu"]
