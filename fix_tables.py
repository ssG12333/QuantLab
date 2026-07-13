"""Fix markdown tables in notebooks: ensure first column has a header label."""
import json, re
from pathlib import Path

def fix_table_in_source(lines):
    """Fix a markdown table: add first-column header if empty, ensure consistent cols."""
    # Find table boundaries within this cell's source lines
    in_table = False
    table_start = None
    fixed = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect table start: line begins with | and has multiple | separators
        if stripped.startswith('|') and stripped.count('|') >= 2:
            if not in_table or (i > 0 and lines[i-1].strip() == ''):
                # This could be a new table
                pass
            in_table = True
            if table_start is None:
                # Check if next line is a separator row (|---|...)
                if i+1 < len(lines) and re.match(r'^\|[\s\-:|]+\|', lines[i+1].strip()):
                    table_start = i
                    header = stripped
                    # Fix: if first column header is empty (starts with '| |'),
                    # add a label like '| 项目 |' or '| |'
                    # Actually: check if first | is followed by space then |
                    if re.match(r'^\|\s*\|', header):
                        # First column is empty - add a placeholder
                        # Replace first '| |' with '|  |' (keep empty but ensure it's recognized)
                        # Actually the issue is some renderers want a non-empty header.
                        # Change to '| 对比项 |' or keep empty
                        fixed_header = header.replace('| |', '| 对比维度 |', 1)
                        if fixed_header != header:
                            lines[i] = line.replace(header.strip(), fixed_header.strip(), 1)
            i += 1
            continue

        if in_table and not stripped.startswith('|') and stripped != '':
            in_table = False
            table_start = None

        i += 1

    return lines


def fix_notebook_tables(nb_path):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    fixes = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'markdown':
            continue

        source = cell['source']
        # Find all tables in this cell and fix first column
        new_source = []
        i = 0
        while i < len(source):
            line = source[i]
            stripped = line.strip()

            # Check if this line starts a table (begins with | and has separator row next)
            if stripped.startswith('|') and stripped.count('|') >= 2:
                if i+1 < len(source) and re.match(r'^\|[\s\-:|]+\|', source[i+1].strip()):
                    # This is a table header row. Check if first column is empty.
                    cols = stripped.split('|')
                    # cols[0] is '' (before first |), cols[1] is first column content
                    if len(cols) >= 2 and cols[1].strip() == '':
                        # First column header is empty - add a placeholder
                        old_line = line
                        line = line.replace('| |', '| 维度 |', 1)
                        if line != old_line:
                            fixes += 1
            new_source.append(line)
            i += 1

        cell['source'] = new_source

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return fixes


if __name__ == '__main__':
    base = Path(r'c:\Users\weijiashengs\Desktop\量化学习o\notebooks')
    total = 0
    for nb_file in sorted(base.glob('*.ipynb')):
        f = fix_notebook_tables(str(nb_file))
        if f > 0:
            print(f'  {nb_file.name}: fixed {f} table(s)')
        total += f
    print(f'\nTotal fixes: {total}')
