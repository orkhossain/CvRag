from typing import List

from langchain_core.documents import Document


def format_docs(documents: List[Document]) -> str:
    if not documents:
        return "No relevant information found."

    lines = []
    seen_content = set()

    sections = {}
    for doc in documents:
        section = doc.metadata.get("section", "other")
        sections.setdefault(section, []).append(doc)

    section_order = [
        "summary",
        "experience",
        "skills",
        "project",
        "education",
        "certification",
        "mentorship",
        "languages",
        "basics",
    ]

    for section in section_order:
        if section in sections:
            for doc in sections[section]:
                content = doc.page_content.strip()
                if content not in seen_content:
                    seen_content.add(content)
                    meta = doc.metadata

                    tag_parts = [section.upper()]
                    if meta.get("company"):
                        tag_parts.append(meta["company"])
                    if meta.get("name"):
                        tag_parts.append(meta["name"])
                    if meta.get("dates"):
                        tag_parts.append(meta["dates"])
                    if meta.get("type"):
                        tag_parts.append(f"({meta['type']})")

                    tag = f"[{' | '.join(tag_parts)}]"
                    lines.append(f"- {tag} {content}")

    for section, docs in sections.items():
        if section not in section_order:
            for doc in docs:
                content = doc.page_content.strip()
                if content not in seen_content:
                    seen_content.add(content)
                    lines.append(f"- [{section.upper()}] {content}")

    return "\n".join(lines[:15])
