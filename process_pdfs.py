#!/usr/bin/env python3
"""
Process PDFs containing HIP/ROCm documentation and distill into Markdown
for ingestion by KANDS knowledge base.
"""
import os
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    print("Installing pypdf...")
    os.system(f"{sys.executable} -m pip install pypdf")
    from pypdf import PdfReader

# PDF files to process
PDF_FILES = [
    r"C:\Users\rr\OneDrive\Desktop\dxl\knowledge\rocm-docs-amd-com-HIP-en-docs-6.1.5.pdf",
    r"C:\Users\rr\OneDrive\Desktop\dxl\knowledge\intro_hip_programming.pdf",
    r"C:\Users\rr\Downloads\HIP_Programming_Guide-texts.pdf"
]

# Output directory in KANDS
OUTPUT_DIR = Path(r"C:\Users\rr\K_And_S\learned")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        reader = PdfReader(pdf_path)
        text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text.append(f"--- Page {i + 1} ---\n{page_text}")
        return "\n\n".join(text)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

def distill_text(raw_text: str, title: str) -> str:
    """
    Distill raw PDF text into structured Markdown.
    Focuses on key concepts, code examples, and API references.
    """
    lines = raw_text.split('\n')

    # Simple distillation: organize by detecting headers and code blocks
    distilled = [f"# {title}\n"]
    distilled.append(f"_Source: Distilled from PDF documentation_\n")

    in_code_block = False
    current_section = None

    for line in lines:
        stripped = line.strip()

        # Skip page markers and empty lines
        if stripped.startswith("--- Page") or not stripped:
            continue

        # Detect potential headers (all caps or followed by colon)
        if stripped.isupper() or (stripped.endswith(':') and len(stripped) < 80):
            distilled.append(f"\n## {stripped.rstrip(':')}\n")
        # Detect code blocks (indented or containing common code patterns)
        elif any(kw in stripped for kw in ['hipMemcpy', 'hipMalloc', 'kernel', '#include', '__global__', 'dim3', 'threadIdx']):
            if not in_code_block:
                distilled.append("\n```cpp\n")
                in_code_block = True
            distilled.append(stripped)
        else:
            if in_code_block:
                distilled.append("\n```\n")
                in_code_block = False
            distilled.append(stripped)

    if in_code_block:
        distilled.append("\n```\n")

    return "\n".join(distilled)

def process_pdf(pdf_path: str, output_dir: Path):
    """Process a single PDF file and save as Markdown."""
    if not os.path.exists(pdf_path):
        print(f"Warning: PDF not found: {pdf_path}")
        return False

    # Generate a filename from the PDF name
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # Clean up for use as a knowledge filename
    safe_name = pdf_name.replace('-', '_').replace(' ', '_').lower()
    md_filename = f"{safe_name}.md"

    print(f"Processing: {pdf_path}")

    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        print(f"  Warning: No text extracted from {pdf_path}")
        return False

    # Count words
    word_count = len(raw_text.split())
    print(f"  Extracted {word_count} words")

    # Distill into structured markdown
    distilled = distill_text(raw_text, pdf_name.replace('_', ' ').title())

    # Save the distilled markdown
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / md_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(distilled)

    print(f"  Saved distilled content to: {output_path}")
    return True

def main():
    print("Processing HIP/ROCm PDF documentation for KANDS knowledge base...")

    output_dir = Path(OUTPUT_DIR)
    successful = 0

    for pdf_path in PDF_FILES:
        if process_pdf(pdf_path, output_dir):
            successful += 1

    print(f"\nProcessed {successful}/{len(PDF_FILES)} PDFs successfully.")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()