from collections import defaultdict
from collections.abc import Iterable

from rag_permission.models import DocumentChunk, ParsedDocument


def split_fixed_window(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    if len(text) <= chunk_size:
        return [text]
    pieces = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        pieces.append(text[start : start + chunk_size])
        if start + chunk_size >= len(text):
            break
    return pieces


def split_recursive(
    text: str, chunk_size: int, separators: Iterable[str] = ("\n\n", "\n", "。", "；", "，", ".")
) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    for separator in separators:
        if separator and separator in text:
            parts = [part for part in text.split(separator) if part]
            pieces: list[str] = []
            current = ""
            for part in parts:
                candidate = part if not current else current + separator + part
                if len(candidate) <= chunk_size:
                    current = candidate
                    continue
                if current:
                    pieces.append(current)
                pieces.extend(split_recursive(part, chunk_size, separators))
                current = ""
            if current:
                pieces.append(current)
            return pieces
    return [text[start : start + chunk_size] for start in range(0, len(text), chunk_size)]


def _element_chunks(
    parsed: ParsedDocument,
    chunks: list[str],
    chunk_type: str,
    element_index: int,
    start_number: int,
    parent_id: str | None = None,
    parent_text: str | None = None,
) -> list[DocumentChunk]:
    element = parsed.elements[element_index]
    return [
        DocumentChunk(
            chunk_id=f"{parsed.doc_id}:{chunk_type}:{start_number + number}",
            doc_id=parsed.doc_id,
            text=text,
            chunk_type=chunk_type,
            section_path=element.section_path,
            source=parsed.source,
            title=parsed.title,
            acl_groups=parsed.acl_groups,
            locator=element.locator,
            parent_id=parent_id,
            parent_text=parent_text,
            metadata={"element_kind": element.kind, "element_index": element_index},
        )
        for number, text in enumerate(chunks, 1)
    ]


def _section_groups(parsed: ParsedDocument):
    groups = defaultdict(list)
    for index, element in enumerate(parsed.elements):
        if element.kind == "heading" and not element.text:
            continue
        groups[element.section_path].append(index)
    return groups


def chunk_document(
    parsed: ParsedDocument,
    strategy: str = "parent_child",
    chunk_size: int = 260,
    overlap: int = 48,
    parent_chunk_size: int = 1000,
) -> list[DocumentChunk]:
    if strategy == "fixed":
        chunks = []
        for index, element in enumerate(parsed.elements):
            pieces = split_fixed_window(element.text, chunk_size, overlap)
            chunks.extend(_element_chunks(parsed, pieces, "fixed", index, len(chunks)))
        return chunks

    if strategy == "recursive":
        chunks = []
        for index, element in enumerate(parsed.elements):
            pieces = split_recursive(element.text, chunk_size)
            chunks.extend(_element_chunks(parsed, pieces, "recursive", index, len(chunks)))
        return chunks

    if strategy == "section":
        chunks = []
        for indices in _section_groups(parsed).values():
            group_text = "\n".join(parsed.elements[index].text for index in indices)
            pieces = split_recursive(group_text, chunk_size, separators=("\n", "。", "；", "，", "."))
            first = parsed.elements[indices[0]]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{parsed.doc_id}:section:{len(chunks) + 1}",
                    doc_id=parsed.doc_id,
                    text=group_text if len(group_text) <= chunk_size else pieces[0],
                    chunk_type="section",
                    section_path=first.section_path,
                    source=parsed.source,
                    title=parsed.title,
                    acl_groups=parsed.acl_groups,
                    locator=first.locator,
                    metadata={"element_indices": list(indices)},
                )
            )
            if len(pieces) > 1:
                for piece in pieces[1:]:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{parsed.doc_id}:section:{len(chunks) + 1}",
                            doc_id=parsed.doc_id,
                            text=piece,
                            chunk_type="section",
                            section_path=first.section_path,
                            source=parsed.source,
                            title=parsed.title,
                            acl_groups=parsed.acl_groups,
                            locator=first.locator,
                            metadata={"element_indices": list(indices)},
                        )
                    )
        return chunks

    if strategy == "parent_child":
        child_overlap = min(overlap, chunk_size - 1)
        chunks = []
        for indices in _section_groups(parsed).values():
            group_text = "\n".join(parsed.elements[index].text for index in indices)
            parents = split_recursive(group_text, parent_chunk_size, separators=("\n", "。", "；", "，", "."))
            element_index = indices[0]
            for parent_number, parent_text in enumerate(parents, 1):
                parent_id = f"{parsed.doc_id}:parent:{len(chunks) + 1}:{parent_number}"
                children = split_fixed_window(parent_text, chunk_size, child_overlap)
                for child_number, child_text in enumerate(children, 1):
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{parent_id}:child:{child_number}",
                            doc_id=parsed.doc_id,
                            text=child_text,
                            chunk_type="child",
                            section_path=parsed.elements[element_index].section_path,
                            source=parsed.source,
                            title=parsed.title,
                            acl_groups=parsed.acl_groups,
                            locator=parsed.elements[element_index].locator,
                            parent_id=parent_id,
                            parent_text=parent_text,
                            metadata={"element_indices": list(indices), "parent_number": parent_number},
                        )
                    )
        return chunks

    raise ValueError(f"Unknown chunking strategy: {strategy}")
