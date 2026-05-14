"""Unit tests for the Feishu markdown-table → code-block conversion helpers."""

import textwrap
import unittest

from gateway.platforms.feishu import (
    _convert_markdown_tables_to_code_blocks,
    _display_width,
    _render_table_as_code_block,
    _split_table_row,
)


class TestDisplayWidth(unittest.TestCase):
    def test_ascii_counts_one_per_char(self):
        self.assertEqual(_display_width("hello"), 5)
        self.assertEqual(_display_width(""), 0)

    def test_cjk_counts_two_per_char(self):
        self.assertEqual(_display_width("中文"), 4)
        self.assertEqual(_display_width("a中b"), 4)

    def test_fullwidth_punct_counts_two(self):
        self.assertEqual(_display_width("！？"), 4)


class TestSplitTableRow(unittest.TestCase):
    def test_basic_row(self):
        self.assertEqual(_split_table_row("| a | b | c |"), ["a", "b", "c"])

    def test_strips_per_cell_whitespace(self):
        self.assertEqual(_split_table_row("|  x  |y|  z|"), ["x", "y", "z"])

    def test_empty_cells_preserved(self):
        self.assertEqual(_split_table_row("| a |  | c |"), ["a", "", "c"])


class TestRenderTableAsCodeBlock(unittest.TestCase):
    def test_basic_alignment(self):
        rendered = _render_table_as_code_block([
            ["Col A", "Col B"],
            ["v1", "foo"],
            ["v22", "longer"],
        ])
        # Column widths: A=max(5,2,3)=5, B=max(5,3,6)=6
        expected = textwrap.dedent(
            """\
            ```
            Col A   Col B
            -----   ------
            v1      foo
            v22     longer
            ```"""
        )
        self.assertEqual(rendered, expected)

    def test_cjk_columns_align(self):
        rendered = _render_table_as_code_block([
            ["名称", "数量"],
            ["苹果", "3"],
            ["香蕉果", "12"],
        ])
        # col0 width = max(名称=4, 苹果=4, 香蕉果=6) = 6
        # col1 width = max(数量=4, 3=1, 12=2) = 4
        # Each row: col0 padded to 6 + "   " sep + col1; rstripped.
        expected = textwrap.dedent(
            """\
            ```
            名称     数量
            ------   ----
            苹果     3
            香蕉果   12
            ```"""
        )
        self.assertEqual(rendered, expected)


class TestConvertMarkdownTablesToCodeBlocks(unittest.TestCase):
    def test_passthrough_when_no_pipes(self):
        text = "hello\n\nworld"
        self.assertEqual(_convert_markdown_tables_to_code_blocks(text), text)

    def test_passthrough_when_pipes_but_no_separator(self):
        # Pipes that aren't a real table (no |---| separator) stay verbatim.
        text = "use `a | b` syntax\nand carry on"
        self.assertEqual(_convert_markdown_tables_to_code_blocks(text), text)

    def test_table_only(self):
        text = textwrap.dedent(
            """\
            | A | B |
            |---|---|
            | 1 | 2 |
            | 3 | 4 |
            """
        )
        result = _convert_markdown_tables_to_code_blocks(text)
        self.assertIn("```\nA   B\n-   -\n1   2\n3   4\n```", result)
        # Trailing newline preserved
        self.assertTrue(result.endswith("\n"))

    def test_table_with_surrounding_prose(self):
        text = textwrap.dedent(
            """\
            Summary:

            | A | B |
            |---|---|
            | 1 | 2 |

            See above.
            """
        )
        result = _convert_markdown_tables_to_code_blocks(text)
        self.assertIn("Summary:", result)
        self.assertIn("See above.", result)
        self.assertIn("```\nA   B\n-   -\n1   2\n```", result)
        # Order is preserved
        self.assertLess(result.index("Summary:"), result.index("```"))
        self.assertLess(result.index("```"), result.index("See above."))

    def test_multiple_tables(self):
        text = textwrap.dedent(
            """\
            First:
            | A | B |
            |---|---|
            | 1 | 2 |

            Second:
            | X | Y |
            |---|---|
            | 9 | 8 |
            """
        )
        result = _convert_markdown_tables_to_code_blocks(text)
        # Two distinct code blocks
        self.assertEqual(result.count("```"), 4)
        self.assertIn("A   B", result)
        self.assertIn("X   Y", result)

    def test_cjk_columns_in_pipeline(self):
        text = textwrap.dedent(
            """\
            报告:
            | 名称 | 数量 |
            |------|------|
            | 苹果 | 3 |
            | 香蕉果 | 12 |
            """
        )
        result = _convert_markdown_tables_to_code_blocks(text)
        self.assertIn("报告:", result)
        self.assertIn("名称", result)
        # Header row appears inside a fenced code block
        self.assertRegex(result, r"```\n名称")

    def test_single_column_table(self):
        text = textwrap.dedent(
            """\
            | only |
            |------|
            | a    |
            | bb   |
            """
        )
        result = _convert_markdown_tables_to_code_blocks(text)
        self.assertIn("```\nonly\n----\na\nbb\n```", result)

    def test_ragged_rows_padded(self):
        text = textwrap.dedent(
            """\
            | A | B | C |
            |---|---|---|
            | 1 | 2 |
            | x | y | z |
            """
        )
        result = _convert_markdown_tables_to_code_blocks(text)
        self.assertIn("A   B   C", result)
        # Missing cell: row pads then rstrips, so 3-col row "x y z" still aligns
        # column-by-column with the 2-col row above.
        self.assertIn("1   2\nx   y   z", result)

    def test_empty_cells_preserved(self):
        text = textwrap.dedent(
            """\
            | A | B |
            |---|---|
            | 1 |   |
            |   | 2 |
            """
        )
        result = _convert_markdown_tables_to_code_blocks(text)
        self.assertIn("A   B", result)
        # Row "1, ''" rstrips to "1"; row "'', 2" keeps leading pad "    2".
        self.assertIn("1\n    2", result)

    def test_table_inside_existing_code_block_is_left_alone(self):
        text = textwrap.dedent(
            """\
            ```
            | A | B |
            |---|---|
            | 1 | 2 |
            ```
            """
        )
        result = _convert_markdown_tables_to_code_blocks(text)
        # Should be unchanged — the table is documentation inside a code block.
        self.assertEqual(result, text)

    def test_table_immediately_after_fence(self):
        text = textwrap.dedent(
            """\
            ```
            code
            ```
            | A | B |
            |---|---|
            | 1 | 2 |
            """
        )
        result = _convert_markdown_tables_to_code_blocks(text)
        # Code block preserved
        self.assertIn("```\ncode\n```", result)
        # Table converted
        self.assertIn("```\nA   B\n-   -\n1   2\n```", result)


class TestBuildOutboundPayloadRouting(unittest.TestCase):
    """Tables now route to post-type (with converted code block), not text."""

    def _adapter(self):
        from gateway.config import PlatformConfig
        from gateway.platforms.feishu import FeishuAdapter
        return FeishuAdapter(PlatformConfig())

    def test_table_content_routes_to_post(self):
        import json
        content = textwrap.dedent(
            """\
            Report:

            | A | B |
            |---|---|
            | 1 | 2 |
            """
        )
        msg_type, payload = self._adapter()._build_outbound_payload(content)
        self.assertEqual(msg_type, "post")
        parsed = json.loads(payload)
        rendered = "\n".join(
            elem["text"] for row in parsed["zh_cn"]["content"] for elem in row
        )
        self.assertIn("```", rendered)
        self.assertIn("A   B", rendered)
        self.assertNotIn("|---|", rendered)

    def test_plain_text_unaffected(self):
        import json
        msg_type, payload = self._adapter()._build_outbound_payload("just hello")
        self.assertEqual(msg_type, "text")
        self.assertEqual(json.loads(payload), {"text": "just hello"})

    def test_pipes_without_table_stay_text(self):
        import json
        msg_type, payload = self._adapter()._build_outbound_payload("a | b without separator")
        self.assertEqual(msg_type, "text")
        self.assertEqual(json.loads(payload), {"text": "a | b without separator"})


if __name__ == "__main__":
    unittest.main()
