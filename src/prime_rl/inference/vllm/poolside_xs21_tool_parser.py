import regex as re
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParserManager
from vllm.tool_parsers.poolside_v1_tool_parser import PoolsideV1ToolParser


@ToolParserManager.register_module("poolside_xs21")
class PoolsideXS21ToolParser(PoolsideV1ToolParser):
    """Parse Laguna XS-2.1 tool calls whose name and first argument are adjacent."""

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.func_detail_regex = re.compile(
            r"<tool_call>\s*([^\n<]*?)(?:\n|(?=<arg_key>)|(?=</tool_call>))"
            r"(.*?)</tool_call>",
            re.DOTALL,
        )
