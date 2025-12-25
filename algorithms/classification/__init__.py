# Classification algorithms module for agricultural zoning framework
from .equal_interval import EqualIntervalClassification
from .natural_breaks import NaturalBreaksClassification

__all__ = ['EqualIntervalClassification', 'NaturalBreaksClassification']