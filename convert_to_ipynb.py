#!/usr/bin/env python3
"""Convert Stage markdown documents to Jupyter notebooks (.ipynb)."""
import json, re, sys, os, io
from pathlib import Path

# Fix Windows GBK encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def split_md_to_cells(text: str) -> list[dict]:
    """Split markdown text into a list of notebook cells."""
    cells = []
    # Split on ```python blocks, keeping the delimiters for context
    # Pattern: everything up to ```python, then the python code, then after the closing ```
    pattern = r'(.*?)```python\s*\n(.*?)```'
    parts = re.split(pattern, text, flags=re.DOTALL)

    # parts[0] = text before first ```python
    # parts[1] = code in first ```python
    # parts[2] = text between first ``` and next ```python
    # parts[3] = code in second ```python
    # parts[4] = text after last ```
    # ... pattern repeats

    i = 0
    while i < len(parts):
        if i % 3 == 0:
            # Markdown text (before first code block, or between code blocks)
            md_text = parts[i].strip()
            if md_text:
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": _normalize_lines(md_text),
                })
            i += 1
        elif i % 3 == 1:
            # Code block content
            code_text = parts[i].strip()
            if code_text:
                cells.append({
                    "cell_type": "code",
                    "metadata": {},
                    "source": _normalize_lines(code_text),
                    "outputs": [],
                    "execution_count": None,
                })
            i += 1
        else:
            # The closing ``` delimiter - skip (i % 3 == 2)
            # But parts[i] might contain text after the closing ``` before the next ```python
            # Actually with re.split, parts[2] should be the text between blocks
            # But our pattern captures (text_before)(code)(text_after)...
            # Let me rethink the pattern
            i += 1

    return cells


def split_md_to_cells_v2(text: str) -> list[dict]:
    """Better approach: find all code blocks and split around them."""
    cells = []
    pos = 0

    # Find all python code blocks
    code_pattern = re.compile(r'```python\s*\n(.*?)```', re.DOTALL)

    for match in code_pattern.finditer(text):
        # Markdown before this code block
        md_before = text[pos:match.start()].strip()
        if md_before:
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": _normalize_lines(md_before),
            })

        # The code block
        code = match.group(1).strip()
        if code:
            cells.append({
                "cell_type": "code",
                "metadata": {},
                "source": _normalize_lines(code),
                "outputs": [],
                "execution_count": None,
            })

        pos = match.end()

    # Remaining markdown after last code block
    md_after = text[pos:].strip()
    if md_after:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": _normalize_lines(md_after),
        })

    return cells


def _normalize_lines(text: str) -> list[str]:
    """Convert text to list of lines with trailing newlines (Jupyter format)."""
    lines = text.split('\n')
    # Add trailing newline to each line except possibly the last
    return [line + '\n' for line in lines[:-1]] + ([lines[-1] + '\n'] if lines else [])


def build_notebook(cells: list[dict]) -> dict:
    """Build a complete notebook JSON."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def convert_file(md_path: str, output_dir: str = None):
    """Convert a single markdown file to ipynb."""
    path = Path(md_path)
    if not path.exists():
        print(f"  SKIP {path.name} — file not found")
        return

    text = path.read_text(encoding='utf-8')
    cells = split_md_to_cells_v2(text)
    notebook = build_notebook(cells)

    if output_dir:
        out_path = Path(output_dir) / path.with_suffix('.ipynb').name
    else:
        out_path = path.with_suffix('.ipynb')

    out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')

    # Count cells
    md_count = sum(1 for c in cells if c['cell_type'] == 'markdown')
    code_count = sum(1 for c in cells if c['cell_type'] == 'code')
    print(f"  OK {out_path.name} — {md_count} md + {code_count} code = {len(cells)} cells")


if __name__ == '__main__':
    base = Path(r'c:\Users\weijiashengs\Desktop\量化学习o')
    docs_dir = base / 'docs'

    # Create output directory for notebooks
    nbs_dir = base / 'notebooks'
    nbs_dir.mkdir(exist_ok=True)

    stages = [
        'Stage0_量化基础与硬件基石',
        'Stage1_PyTorch QAT PT2E 深度拆解',
        'Stage1.5_QAT训练深度剖析',
        'Stage2_LSQ与可微量化参数',
        'Stage3_PTQ进阶算法',
    ]

    print("Converting markdown to Jupyter notebooks:\n")
    for stage in stages:
        md_path = docs_dir / f'{stage}.md'
        convert_file(str(md_path), str(nbs_dir))

    print(f"\nDone! Notebooks saved to: {nbs_dir}")
