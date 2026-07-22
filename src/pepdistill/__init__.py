"""pepdistill: distill AlphaPeptDeep into fast, hardware-friendly spectral libraries."""

__version__ = "0.1.0"

from .chem import Peptide

__all__ = ["Peptide", "__version__"]
