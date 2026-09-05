import tempfile
from pathlib import Path

from rag_permission.ingest import chunk_document, parse_document


SAMPLE = """# 设备手册
## 故障码
E-1002 是风扇故障，需要检查电源。
| 故障码 | 含义 |
| --- | --- |
| E-1002 | 风扇故障 |
"""


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.md"
        path.write_text(SAMPLE, encoding="utf-8")
        parsed = parse_document(path, ("public",))
        print(f"doc_id={parsed.doc_id}, elements={len(parsed.elements)}")
        for strategy in ("fixed", "recursive", "section", "parent_child"):
            chunks = chunk_document(parsed, strategy)
            print(
                f"{strategy}: {len(chunks)} chunks, "
                f"first_id={chunks[0].chunk_id}, acl={chunks[0].acl_groups}"
            )


if __name__ == "__main__":
    main()
