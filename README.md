# convertdocx2md
Script to convert reports in Word .docx to markdown .md format


## Single file
python ssaf_docx_to_md.py ssaf_report_Alder-W02.docx \
  -o ssaf_report_Alder-W02.md \
  --media-dir ssaf_report_Alder-W02_media
``

## Batch folder
python ssaf_docx_to_md.py ./reports \
  --out-dir ./md \
  --media-root ./md_media
``

## Output structure (what you’ll get)
For each report:
- md/<report>.md (normalized, GraphRAG-ready)
- md_media/<report>/... (extracted plots)
