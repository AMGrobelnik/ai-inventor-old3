"""
Base class for all aii_lib tools.

All aii_lib tools should inherit from AIITool. This creates a common base class
that ToolUniverse uses to group tools under the "AIITool" category in listings.
"""

from tooluniverse.base_tool import BaseTool


class AIITool(BaseTool):
    """
    Base class for all aii_lib custom tools.

    ToolUniverse groups tools by their parent class type. By having all aii_lib
    tools inherit from AIITool (instead of directly from BaseTool), they will
    be grouped together under the "AIITool" category in tool listings.

    Usage:
        from aii_lib.abilities.tools.base import AIITool

        @register_tool('MyTool', config={...})
        class MyTool(AIITool):
            def run(self, arguments=None, **kwargs):
                ...
    """
    pass
