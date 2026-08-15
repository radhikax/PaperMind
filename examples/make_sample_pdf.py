from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

out = "examples/sample.pdf"

c = canvas.Canvas(out, pagesize=letter)
width, height = letter
c.setFont("Helvetica", 14)
c.drawString(72, height - 72, "Sample Paper")
c.setFont("Helvetica", 10)
text = c.beginText(72, height - 100)
for i in range(1, 40):
    text.textLine(f"This is sample paragraph line {i}. It contains some example content for testing PDF ingestion.")
c.drawText(text)
c.showPage()
c.save()
print(f"Wrote {out}")
