import fitz

doc = fitz.open('report/ML_Assignment_Report.pdf')
print("=== PDF REPORT AUDIT ===")
print("Total Pages: {} (Limit: maximum 4-5 pages)".format(len(doc)))
for i, page in enumerate(doc):
    text = page.get_text().strip()
    first_line = text.split('\n')[0] if text else "Empty"
    print("Page {}: {} characters | Starts with: '{}'".format(i + 1, len(text), first_line[:50]))
