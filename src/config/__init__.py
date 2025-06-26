"""
Configuration package for the facial recognition system.

This package contains all configuration-related code including settings management
and logging configuration.
"""

from .settings import Settings, get_settings
from .logging import setup_logging

__all__ = ['Settings', 'get_settings', 'setup_logging']
