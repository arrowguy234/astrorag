"""
AstroRAG pipeline stages subpackage.

Each stage is implemented as a class with a run() method that takes
the corpus data and produces a specific output. Stages are composed
by run_pipeline.py.
"""

from astrorag.stages.stage0_decompose import Stage0Decompose, decompose_query

__all__ = [
    "Stage0Decompose",
    "decompose_query",
]
"""
AstroRAG pipeline stages subpackage.
"""

from astrorag.stages.stage0_decompose import Stage0Decompose, decompose_query
from astrorag.stages.stage1_bm25       import Stage1BM25

__all__ = [
    "Stage0Decompose",
    "decompose_query",
    "Stage1BM25",
]
"""
AstroRAG pipeline stages subpackage.
"""

from astrorag.stages.stage0_decompose import Stage0Decompose, decompose_query
from astrorag.stages.stage1_bm25       import Stage1BM25
from astrorag.stages.stage2_graph      import Stage2Graph

__all__ = [
    "Stage0Decompose",
    "decompose_query",
    "Stage1BM25",
    "Stage2Graph",
]
"""AstroRAG pipeline stages subpackage."""

from astrorag.stages.stage0_decompose import Stage0Decompose, decompose_query
from astrorag.stages.stage1_bm25       import Stage1BM25
from astrorag.stages.stage2_graph      import Stage2Graph
from astrorag.stages.stage3_rerank     import Stage3Rerank, Stage3Result

__all__ = [
    "Stage0Decompose",
    "decompose_query",
    "Stage1BM25",
    "Stage2Graph",
    "Stage3Rerank",
    "Stage3Result",
]
"""AstroRAG pipeline stages subpackage."""

from astrorag.stages.stage0_decompose import Stage0Decompose, decompose_query
from astrorag.stages.stage1_bm25       import Stage1BM25
from astrorag.stages.stage2_graph      import Stage2Graph
from astrorag.stages.stage3_rerank     import Stage3Rerank, Stage3Result
from astrorag.stages.stage4_pdf        import Stage4PDF

__all__ = [
    "Stage0Decompose",
    "decompose_query",
    "Stage1BM25",
    "Stage2Graph",
    "Stage3Rerank",
    "Stage3Result",
    "Stage4PDF",
]
"""AstroRAG pipeline stages subpackage."""

from astrorag.stages.stage0_decompose import Stage0Decompose, decompose_query
from astrorag.stages.stage1_bm25       import Stage1BM25
from astrorag.stages.stage2_graph      import Stage2Graph
from astrorag.stages.stage3_rerank     import Stage3Rerank, Stage3Result
from astrorag.stages.stage4_pdf        import Stage4PDF
from astrorag.stages.stage5_summarise  import Stage5Summarise, Stage5Result

__all__ = [
    "Stage0Decompose",
    "decompose_query",
    "Stage1BM25",
    "Stage2Graph",
    "Stage3Rerank",
    "Stage3Result",
    "Stage4PDF",
    "Stage5Summarise",
    "Stage5Result",
]