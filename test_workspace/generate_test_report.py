from pathlib import Path

out = Path(r"C:\Users\Ricky\Documents\Project\XSpoonAi\spoon-bot\test_workspace\test_report.pdf")

header = "Test Report"
lines = [
    "Line 1: Dynamic tool loading test",
    "Line 2: Skills are working correctly",
    "Line 3: PDF export via document_export skill",
]

objects = []

def add_obj(content: str):
    objects.append(content)

# PDF structure
add_obj("<< /Type /Catalog /Pages 2 0 R >>")
add_obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
add_obj("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>")
add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

# Content stream
cmds = [
    "BT",
    "/F1 24 Tf",
    "72 740 Td",
    f"({header}) Tj",
    "0 -40 Td",
    "/F1 12 Tf",
]
for line in lines:
    safe = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
    cmds.append(f"({safe}) Tj")
    cmds.append("0 -22 Td")
cmds.append("ET")

stream = "\n".join(cmds)
add_obj(f"<< /Length {len(stream.encode('latin-1'))} >>\nstream\n{stream}\nendstream")

# Assemble bytes
pdf = bytearray()
pdf.extend(b"%PDF-1.4\n")

offsets = [0]
for i, obj in enumerate(objects, start=1):
    offsets.append(len(pdf))
    pdf.extend(f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1"))

xref_pos = len(pdf)
size = len(objects) + 1
pdf.extend(f"xref\n0 {size}\n".encode("latin-1"))
pdf.extend(b"0000000000 65535 f \n")
for off in offsets[1:]:
    pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))

pdf.extend(f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1"))

out.write_bytes(pdf)
print(f"Created: {out}")
print(f"Size: {out.stat().st_size} bytes")
