"""Generate PDF report from Markdown using markdown + basic HTML-to-PDF."""
import os

md_file = os.path.join('report', 'ML_Assignment_Report.md')
pdf_file = os.path.join('report', 'ML_Assignment_Report.pdf')
html_file = os.path.join('report', 'ML_Assignment_Report.html')

# Read markdown
with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert to HTML
try:
    import markdown
    html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
except ImportError:
    # Simple fallback
    import re
    html_body = md_content.replace('\n', '<br>\n')

html_full = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>ML Assignment 1 Report - BT2024039</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px 60px; font-size: 11pt; line-height: 1.5; color: #333; }}
    h1 {{ color: #1a1a1a; font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 5px; }}
    h2 {{ color: #2c2c2c; font-size: 14pt; margin-top: 20px; border-bottom: 1px solid #999; padding-bottom: 3px; }}
    h3 {{ color: #444; font-size: 12pt; margin-top: 15px; }}
    table {{ border-collapse: collapse; margin: 10px 0; width: 100%; font-size: 10pt; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
    th {{ background-color: #f0f0f0; font-weight: bold; }}
    tr:nth-child(even) {{ background-color: #fafafa; }}
    code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 10pt; }}
    pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
    strong {{ color: #1a1a1a; }}
    hr {{ border: none; border-top: 1px solid #ddd; margin: 15px 0; }}
    p {{ margin: 5px 0; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_full)

print(f"HTML report saved to: {html_file}")
print("To convert to PDF:")
print("  - Open the HTML file in Chrome/Edge and use 'Print to PDF'")
print("  - Or install wkhtmltopdf and run: wkhtmltopdf report/ML_Assignment_Report.html report/ML_Assignment_Report.pdf")
