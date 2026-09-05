import json
from pathlib import Path

from rag_permission.ingest import chunk_document, parse_document
from rag_permission.models import ParsedDocument, ParsedElement


def _parsed(path: str):
    return parse_document(Path("fixtures") / path, ("public",))


def test_markdown_maintains_section_path_and_error_code():
    parsed = _parsed("sample.md")
    row = next(item for item in parsed.elements if "E-1002" in item.text)
    assert row.section_path == ("公共设备手册", "通用故障码")


def test_docx_keeps_paragraph_and_table_order():
    parsed = _parsed("sample.docx")
    kinds = [element.kind for element in parsed.elements]
    assert kinds.index("heading") < kinds.index("table_row")
    assert any("E001" in element.text for element in parsed.elements if element.kind == "table_row")


def test_pdf_extracts_table_row():
    parsed = _parsed("sample.pdf")
    assert any(element.kind == "table_row" and "劳动法规" in element.text for element in parsed.elements)


def test_fixed_window_overlap_and_id():
    parsed = ParsedDocument(
        doc_id="doc",
        title="manual",
        source="sample.md",
        acl_groups=("public",),
        elements=(ParsedElement("paragraph", "0123456789abcdefg", ("manual",), "L1"),),
    )
    chunks = chunk_document(parsed, "fixed", chunk_size=10, overlap=3)
    assert chunks[0].chunk_id.endswith(":fixed:1")
    assert chunks[0].text[-3:] in chunks[1].text


def test_recursive_chunk_size():
    parsed = _parsed("sample.md")
    chunks = chunk_document(parsed, "recursive", chunk_size=8)
    assert all(len(chunk.text) <= 8 for chunk in chunks)


def test_section_chunking_does_not_cross_sections():
    parsed = _parsed("sample.md")
    chunks = chunk_document(parsed, "section", chunk_size=1000)
    assert {chunk.section_path for chunk in chunks} >= {
        ("公共设备手册", "通用故障码"),
        ("公共设备手册", "保养要求"),
    }


def test_parent_child_uses_small_retrieval_and_large_generation_context():
    parsed = _parsed("sample.md")
    children = chunk_document(parsed, "parent_child", chunk_size=40, parent_chunk_size=200)
    assert all(child.chunk_type == "child" for child in children)
    assert all(child.parent_text and len(child.parent_text) >= len(child.text) for child in children)


def test_chunk_id_is_stable_and_acl_is_inherited():
    parsed = _parsed("sample.md")
    first = chunk_document(parsed, "section")
    second = chunk_document(_parsed("sample.md"), "section")
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.acl_groups == ("public",) for chunk in first)
