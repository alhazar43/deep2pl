"""
Model Factory for creating DeepIRT models.
Allows easy switching between original and optimized implementations.
"""

from .model import DeepIRTModel
from .model_optim import OptimizedDeepIRTModel


def create_model(model_type='optimized', **kwargs):
    """
    Factory function to create DeepIRT models.
    
    Args:
        model_type (str): 'original' or 'optimized' (default)
        **kwargs: Arguments to pass to model constructor
        
    Returns:
        DeepIRTModel or OptimizedDeepIRTModel instance
    """
    if model_type.lower() == 'original':
        return DeepIRTModel(**kwargs)
    elif model_type.lower() == 'optimized':
        return OptimizedDeepIRTModel(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'original' or 'optimized'")


def get_model_class(model_type='optimized'):
    """
    Get the model class without instantiating.
    
    Args:
        model_type (str): 'original' or 'optimized' (default)
        
    Returns:
        Model class
    """
    if model_type.lower() == 'original':
        return DeepIRTModel
    elif model_type.lower() == 'optimized':
        return OptimizedDeepIRTModel
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'original' or 'optimized'")