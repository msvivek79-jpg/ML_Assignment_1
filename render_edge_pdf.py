import os
import subprocess
import time
import fitz

html_file = os.path.abspath(os.path.join('report', 'ML_Assignment_Report.html'))
pdf_file = os.path.abspath(os.path.join('report', 'ML_Assignment_Report.pdf'))

edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(edge_path):
    edge_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

print(f"Edge path: {edge_path}")
print(f"HTML source: {html_file}")
print(f"PDF target: {pdf_file}")

file_url = f"file:///{html_file.replace(os.sep, '/')}"

cmd = [
    edge_path,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_file}",
    file_url
]

subprocess.run(cmd, check=True)
time.sleep(1)

doc = fitz.open(pdf_file)
print(f"\n==========================================")
print(f"SUCCESS: Rendered PDF has {len(doc)} pages.")
print(f"==========================================")

for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    img_path = os.path.join('report', f'page_{i+1}.png')
    pix.save(img_path)
    print(f"Page {i+1} saved to {img_path} ({pix.width}x{pix.height})")
