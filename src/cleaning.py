import json
import re
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MINERU_DIR = Path(
    r"D:\Langchain\pdf_rag_q&a\mineru_output\attention\hybrid_auto"
)

INPUT_FILE = MINERU_DIR / "structured_elements.json"
OUTPUT_FILE = MINERU_DIR / "cleaned_elements.json"


# ============================================================
# EXPECTED CONTENT TYPES
# ============================================================

VALID_CONTENT_TYPES = {
    "document_title",
    "heading",
    "text",
    "equation",
    "figure",
    "table",
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Perform conservative cleaning on normal text.

    Important:
    We intentionally do NOT perform aggressive mathematical
    normalization because equations may contain meaningful
    spacing/symbols.
    """

    if not text:
        return ""

    # Convert common HTML superscript/subscript tags
    text = re.sub(r"<sup>(.*?)</sup>", r"\1", text)
    text = re.sub(r"<sub>(.*?)</sub>", r"\1", text)

    # Remove other simple HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive newlines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


# ============================================================
# VALIDATION
# ============================================================

def validate_element(element, index):
    """
    Validate one structured element.

    Returns:
        list[str]: warnings/errors
    """

    problems = []

    required_keys = [
        "page",
        "content_type",
        "text",
    ]

    # Check required keys
    for key in required_keys:
        if key not in element:
            problems.append(
                f"Missing required key '{key}'"
            )

    # Stop if required information is missing
    if "content_type" not in element:
        return problems

    content_type = element["content_type"]

    # Validate content type
    if content_type not in VALID_CONTENT_TYPES:
        problems.append(
            f"Unknown content_type: {content_type}"
        )

    # Validate page
    page = element.get("page")

    if not isinstance(page, int) or page < 1:
        problems.append(
            f"Invalid page number: {page}"
        )

    # Text validation
    text = element.get("text", "")

    if content_type != "table" and not text.strip():
        problems.append(
            "Empty text"
        )

    # Equation validation
    if content_type == "equation":

        if not text.strip():
            problems.append(
                "Equation has no mathematical text"
            )

        if "image_path" not in element:
            problems.append(
                "Equation has no image_path"
            )

    # Figure validation
    if content_type == "figure":

        if "image_path" not in element:
            problems.append(
                "Figure has no image_path"
            )

    # Table validation
    if content_type == "table":

        if not element.get("html"):
            problems.append(
                "Table has no HTML representation"
            )

        if "image_path" not in element:
            problems.append(
                "Table has no image_path"
            )

    return problems


# ============================================================
# IMAGE PATH VALIDATION
# ============================================================

def validate_image_path(element):
    """
    Check whether referenced MinerU image actually exists.
    """

    image_path = element.get("image_path")

    if not image_path:
        return None

    image_file = MINERU_DIR / image_path

    if not image_file.exists():
        return (
            f"Image does not exist: {image_file}"
        )

    return None


# ============================================================
# CLEAN ONE ELEMENT
# ============================================================

def clean_element(element):
    """
    Clean one structured element while preserving
    document structure and metadata.
    """

    cleaned = element.copy()

    content_type = cleaned.get("content_type")

    # Clean normal textual content
    if "text" in cleaned:
        cleaned["text"] = clean_text(
            cleaned["text"]
        )

    # Clean section names
    if cleaned.get("section"):
        cleaned["section"] = clean_text(
            cleaned["section"]
        )

    if cleaned.get("subsection"):
        cleaned["subsection"] = clean_text(
            cleaned["subsection"]
        )

    # Clean table caption
    if cleaned.get("caption"):
        cleaned["caption"] = clean_text(
            cleaned["caption"]
        )

    # IMPORTANT:
    # Do not modify equation text aggressively.
    if content_type == "equation":
        cleaned["text"] = cleaned.get(
            "text", ""
        ).strip()

    return cleaned


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("STRUCTURED DOCUMENT CLEANING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    print(f"\nLoading:")
    print(INPUT_FILE)

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        elements = json.load(f)

    print(
        f"Loaded {len(elements)} elements"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    content_type_counts = {}

    warnings = []

    cleaned_elements = []

    # --------------------------------------------------------
    # Process elements
    # --------------------------------------------------------

    for index, element in enumerate(elements):

        content_type = element.get(
            "content_type",
            "unknown"
        )

        # Count content types
        content_type_counts[content_type] = (
            content_type_counts.get(
                content_type, 0
            ) + 1
        )

        # Validate
        problems = validate_element(
            element,
            index
        )

        for problem in problems:

            warnings.append(
                f"Element {index} "
                f"(page={element.get('page')}): "
                f"{problem}"
            )

        # Validate image paths
        image_problem = validate_image_path(
            element
        )

        if image_problem:
            warnings.append(
                f"Element {index}: "
                f"{image_problem}"
            )

        # Clean
        cleaned = clean_element(
            element
        )

        cleaned_elements.append(
            cleaned
        )

    # --------------------------------------------------------
    # Save cleaned JSON
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            cleaned_elements,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Print report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CONTENT TYPE SUMMARY")
    print("=" * 60)

    for content_type, count in sorted(
        content_type_counts.items()
    ):

        print(
            f"{content_type:20} : {count}"
        )

    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    if warnings:

        print(
            f"\nFound {len(warnings)} warnings:\n"
        )

        for warning in warnings[:30]:
            print("⚠", warning)

        if len(warnings) > 30:
            print(
                f"\n... and "
                f"{len(warnings) - 30} more"
            )

    else:

        print("\n✓ No validation problems found.")

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)

    print(
        f"\nSaved cleaned document to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()