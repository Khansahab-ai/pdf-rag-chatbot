import json
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

MINERU_DIR = Path(
    r"D:\Langchain\pdf_rag_q&a\mineru_output\attention\hybrid_auto"
)

JSON_FILE = MINERU_DIR / "attention_content_list_v2.json"


# ============================================================
# 2. LOAD MINERU JSON
# ============================================================

if not JSON_FILE.exists():
    raise FileNotFoundError(
        f"MinerU JSON file not found:\n{JSON_FILE}"
    )

with open(JSON_FILE, "r", encoding="utf-8") as f:
    pages = json.load(f)

print("=" * 80)
print("MINERU DOCUMENT")
print("=" * 80)

print(f"JSON file : {JSON_FILE}")
print(f"Pages     : {len(pages)}")

# ============================================================
# 3. TEXT EXTRACTION HELPERS
# ============================================================

def extract_text_items(items):
    """
    Extract normal text from a list of MinerU content items.

    Example:

    [
        {"type": "text", "content": "Hello"},
        {"type": "equation_inline", "content": "x = 1"},
        {"type": "text", "content": "world"}
    ]

    returns:

    "Hello world"
    """

    if not isinstance(items, list):
        return ""

    texts = []

    for item in items:

        if not isinstance(item, dict):
            continue

        if item.get("type") == "text":

            text = item.get("content", "")

            if text:
                texts.append(text)

    return " ".join(texts).strip()


def extract_content_text(content):
    """
    Extract normal text from MinerU content dictionaries.

    This is mainly useful for:
        paragraph
        title
        captions
        footnotes
    """

    if not isinstance(content, dict):
        return ""

    all_text = []

    for key, value in content.items():

        if not isinstance(value, list):
            continue

        text = extract_text_items(value)

        if text:
            all_text.append(text)

    return " ".join(all_text).strip()


# ============================================================
# 4. PARAGRAPH EXTRACTION
# ============================================================

def extract_paragraph(content):
    """
    Extract paragraph content while preserving inline equations.

    Example MinerU:

    text
    equation_inline
    text
    equation_inline

    becomes:

    normal text [EQUATION: x = ...] normal text
    """

    if not isinstance(content, dict):
        return ""

    paragraph_content = content.get(
        "paragraph_content",
        []
    )

    if not isinstance(paragraph_content, list):
        return ""

    parts = []

    for item in paragraph_content:

        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        # ------------------------------------------
        # Normal text
        # ------------------------------------------

        if item_type == "text":

            text = item.get("content", "")

            if text:
                parts.append(text)

        # ------------------------------------------
        # Inline equation
        # ------------------------------------------

        elif item_type == "equation_inline":

            equation = item.get("content", "")

            if equation:
                parts.append(
                    f"[EQUATION: {equation}]"
                )

    return " ".join(parts).strip()


# ============================================================
# 5. TITLE EXTRACTION
# ============================================================

def extract_title(content):
    """
    Extract title text and title level.
    """

    if not isinstance(content, dict):
        return "", None

    title_content = content.get(
        "title_content",
        []
    )

    level = content.get("level")

    text = extract_text_items(title_content)

    return text, level


# ============================================================
# 6. EQUATION EXTRACTION
# ============================================================

def extract_equation(content):
    """
    Extract display/interline equation.

    MinerU stores the actual mathematical expression
    inside:

        content["math_content"]
    """

    if not isinstance(content, dict):
        return ""

    equation = content.get(
        "math_content",
        ""
    )

    return equation.strip()


# ============================================================
# 7. IMAGE / FIGURE EXTRACTION
# ============================================================

def extract_image(content):
    """
    Extract image path and caption.
    """

    if not isinstance(content, dict):
        return "", ""

    # ------------------------------------------
    # Image path
    # ------------------------------------------

    image_source = content.get(
        "image_source",
        {}
    )

    image_path = ""

    if isinstance(image_source, dict):

        image_path = image_source.get(
            "path",
            ""
        )

    # ------------------------------------------
    # Caption
    # ------------------------------------------

    caption = extract_text_items(
        content.get(
            "image_caption",
            []
        )
    )

    return image_path, caption


# ============================================================
# 8. TABLE EXTRACTION
# ============================================================

def extract_table(content):
    """
    Extract table information.

    MinerU gives us:

        table_caption
        html
        table_type
        image_source
    """

    if not isinstance(content, dict):
        return {
            "caption": "",
            "html": "",
            "image_path": "",
            "table_type": ""
        }

    # ------------------------------------------
    # Caption
    # ------------------------------------------

    caption = extract_text_items(
        content.get(
            "table_caption",
            []
        )
    )

    # ------------------------------------------
    # HTML
    # ------------------------------------------

    html = content.get(
        "html",
        ""
    )

    # ------------------------------------------
    # Image
    # ------------------------------------------

    image_source = content.get(
        "image_source",
        {}
    )

    image_path = ""

    if isinstance(image_source, dict):

        image_path = image_source.get(
            "path",
            ""
        )

    # ------------------------------------------
    # Table type
    # ------------------------------------------

    table_type = content.get(
        "table_type",
        ""
    )

    return {
        "caption": caption.strip(),
        "html": html.strip(),
        "image_path": image_path,
        "table_type": table_type
    }


# ============================================================
# 9. SECTION MANAGEMENT
# ============================================================

document_title = None

current_section = None
current_subsection = None


# ============================================================
# 10. TYPES WE DON'T WANT IN RAG
# ============================================================

IGNORED_TYPES = {
    "page_aside_text",
    "page_footer",
    "page_number",
}


# ============================================================
# 11. STRUCTURED DOCUMENT
# ============================================================

structured_elements = []


# ============================================================
# 12. PROCESS EVERY PAGE
# ============================================================

for page_number, page in enumerate(
    pages,
    start=1
):

    for element in page:

        if not isinstance(element, dict):
            continue

        element_type = element.get(
            "type",
            "unknown"
        )

        # ------------------------------------------
        # Ignore irrelevant page elements
        # ------------------------------------------

        if element_type in IGNORED_TYPES:
            continue

        content = element.get(
            "content",
            {}
        )

        bbox = element.get(
            "bbox",
            []
        )


        # ==================================================
        # TITLE
        # ==================================================

        if element_type == "title":

            title_text, level = extract_title(
                content
            )

            if not title_text:
                continue

            # ------------------------------------------
            # Document title
            # ------------------------------------------

            if (
                level == 1
                and document_title is None
            ):

                document_title = title_text

                structured_elements.append({
                    "page": page_number,
                    "content_type": "document_title",
                    "text": title_text,
                    "section": None,
                    "subsection": None,
                    "heading_level": level,
                    "bbox": bbox,
                })

                continue

            # ------------------------------------------
            # Main section
            # ------------------------------------------

            if level == 2:

                current_section = title_text
                current_subsection = None

            # ------------------------------------------
            # Subsection
            # ------------------------------------------

            elif level >= 3:

                current_subsection = title_text

            structured_elements.append({
                "page": page_number,
                "content_type": "heading",
                "text": title_text,
                "section": current_section,
                "subsection": current_subsection,
                "heading_level": level,
                "bbox": bbox,
            })

            continue


        # ==================================================
        # PARAGRAPH
        # ==================================================

        if element_type == "paragraph":

            text = extract_paragraph(
                content
            )

            if not text:
                continue

            structured_elements.append({
                "page": page_number,
                "content_type": "text",
                "text": text,
                "section": current_section,
                "subsection": current_subsection,
                "heading_level": None,
                "bbox": bbox,
            })

            continue


        # ==================================================
        # DISPLAY EQUATION
        # ==================================================

        if element_type == "equation_interline":

            equation = extract_equation(
                content
            )

            if not equation:
                continue

            image_source = content.get(
                "image_source",
                {}
            )

            image_path = ""

            if isinstance(image_source, dict):
                image_path = image_source.get(
                    "path",
                    ""
                )

            structured_elements.append({
                "page": page_number,
                "content_type": "equation",
                "text": equation,
                "section": current_section,
                "subsection": current_subsection,
                "heading_level": None,
                "image_path": image_path,
                "bbox": bbox,
            })

            continue


        # ==================================================
        # FIGURE / IMAGE
        # ==================================================

        if element_type in {
            "image",
            "chart"
        }:

            image_path, caption = extract_image(
                content
            )

            # Some MinerU objects may have no caption.
            text = caption

            structured_elements.append({
                "page": page_number,
                "content_type": "figure",
                "text": text,
                "caption": caption,
                "section": current_section,
                "subsection": current_subsection,
                "heading_level": None,
                "image_path": image_path,
                "bbox": bbox,
            })

            continue


        # ==================================================
        # TABLE
        # ==================================================

        if element_type == "table":

            table = extract_table(
                content
            )

            structured_elements.append({
                "page": page_number,
                "content_type": "table",
                "text": table["caption"],
                "caption": table["caption"],
                "html": table["html"],
                "table_type": table["table_type"],
                "image_path": table["image_path"],
                "section": current_section,
                "subsection": current_subsection,
                "heading_level": None,
                "bbox": bbox,
            })

            continue


# ============================================================
# 13. BASIC STATISTICS
# ============================================================

print()
print("=" * 80)
print("STRUCTURED DOCUMENT STATISTICS")
print("=" * 80)

print(
    f"Document title : {document_title}"
)

print(
    f"Total elements : {len(structured_elements)}"
)


type_counts = {}

for element in structured_elements:

    content_type = element[
        "content_type"
    ]

    type_counts[content_type] = (
        type_counts.get(
            content_type,
            0
        ) + 1
    )


for content_type, count in type_counts.items():

    print(
        f"{content_type:20} : {count}"
    )


# ============================================================
# 14. SAVE STRUCTURED JSON
# ============================================================

OUTPUT_FILE = (
    MINERU_DIR /
    "structured_elements.json"
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        structured_elements,
        f,
        ensure_ascii=False,
        indent=2
    )


print()
print("=" * 80)
print("OUTPUT")
print("=" * 80)

print(
    f"Saved structured data to:\n{OUTPUT_FILE}"
)