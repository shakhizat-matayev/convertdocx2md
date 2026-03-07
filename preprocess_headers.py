"""
Preprocesses Markdown well reports and profiles for GraphRAG indexing by injecting 
well-specific context into all sub-headers.

PURPOSE:
GraphRAG's token-based chunking can cause 'identity loss' when a document is split 
into smaller text units. This script ensures that every chunk retains its association 
with the specific Well_ID by prepending the well name to every H2 and H3 header 
(e.g., '## ASH-W07 - Section 3: Anomaly Detection').

KEY FEATURES:
- Identity Extraction: Automatically identifies the Well_ID from the file's H1 header 
  (e.g., 'Well Health Profile: ASH-W07') or the filename.
- UPPERCASE Enforcement: Forces all Well_IDs to uppercase to ensure perfect entity 
  resolution and prevent duplicate nodes in the Knowledge Graph.
- Re-run Safety: Uses regex to strip existing prefixes before injecting new ones, 
  preventing header nesting (e.g., 'ASH-W07 - ASH-W07 - ...').
- In-Place Updates: Directly modifies files within the GraphRAG 'input/' folder 
  to prepare them for the final 'graphrag index' command.

USAGE:
1. Place all .md files (SSAF reports and Well Profiles) in the 'input/' directory.
2. Run: python preprocess_headers.py
3. Execute: python -m graphrag index --root .
"""

import os
import re

# Target the active GraphRAG input directory
input_folder = 'input'

def inject_well_context(content, well_name):
    """Prepends the Uppercase Well ID to H2 and H3 headers."""
    well_name = well_name.upper().strip()
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        # Target H2 (##) and H3 (###) headers
        if line.startswith('##') and not line.startswith('#####'):
            # Strip any existing well prefixes or 'SSAF' tags to prevent duplicates
            clean_line = re.sub(r'^#+\s*(SSAF|WELL|REPORT|[\w-]+)\s*-\s*', '', line, flags=re.IGNORECASE).strip()
            
            header_level = re.match(r'#+', line).group()
            new_lines.append(f"{header_level} {well_name} - {clean_line}")
        else:
            new_lines.append(line)
            
    return '\n'.join(new_lines)

def extract_well_id(content, filename):
    """
    Finds the Well ID by looking for the last alphanumeric group in the H1 
    or using a specific pattern match.
    """
    # Look for H1 headers like '# SSAF Well Health Report: ASH-W07' 
    # and capture ONLY the part after the colon or at the end.
    h1_match = re.search(r'^#.*[:\s]([\w-]+)$', content, re.MULTILINE)
    
    if h1_match:
        found_id = h1_match.group(1).upper()
        # If it accidentally grabbed 'REPORT' or 'SSAF', fallback to filename
        if found_id in ['REPORT', 'SSAF', 'MD']:
            return filename.split('_')[-1].replace('.md', '').upper()
        return found_id
    
    # Fallback: Extract from filename (e.g., 'ssaf_report_Ash-W07.md')
    return filename.split('_')[-1].replace('.md', '').upper()

# Process the 77 files in the input folder
if os.path.exists(input_folder):
    processed_count = 0
    for filename in os.listdir(input_folder):
        if filename.endswith('.md'):
            file_path = os.path.join(input_folder, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            # Step 1: Extract the actual Well ID (e.g., ASH-W07)
            well_id = extract_well_id(raw_content, filename)

            # Step 2: Inject Uppercase Context into Sub-Headers
            fortified_content = inject_well_context(raw_content, well_id)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fortified_content)
            
            processed_count += 1

    print(f"Success: {processed_count} files updated with the correct Well ID.")
else:
    print(f"Error: Folder '{input_folder}' not found.")