"""Utility functions for aii_pipeline"""

from .pipeline_config import PipelineConfig, rel_path, get_project_root
from .format_helpers import format_value, get_model_short
from aii_lib.prompts import LLMPromptModel
# Private import — for ad-hoc/non-model YAML formatting only.
# For model data, use model.to_prompt_yaml() instead.
from aii_lib.prompts.prompt_format import to_prompt_yaml, to_prompt_yaml_list
from .module_output import init_cumulative, build_module_output, emit_module_output

# Re-export from aii_lib's unified cache (avoid duplicate caches)
from aii_lib.abilities.tools.utils import (
    WebCache,
    search_cache,
    content_cache,
    clear_all_caches,
)

__all__ = [
    'PipelineConfig',
    'rel_path',
    'get_project_root',
    'WebCache',
    'search_cache',
    'content_cache',
    'clear_all_caches',
    # Format helpers
    'format_value',
    'get_model_short',
    'to_prompt_yaml',
    'to_prompt_yaml_list',
    'LLMPromptModel',
    # Module output
    'init_cumulative',
    'build_module_output',
    'emit_module_output',
]
