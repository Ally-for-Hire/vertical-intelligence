"""inline the figures into the overview template -> one self-contained HTML file that can be emailed"""
import base64, os, re
H = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(H, "..", "..", "results", "figures")
t = open(os.path.join(H, "overview_template.html"), encoding="utf-8").read()


def img(m):
    name = m.group(1)
    b = base64.b64encode(open(os.path.join(FIG, name), "rb").read()).decode()
    return f'<img alt="{name}" src="data:image/png;base64,{b}">'


out = re.sub(r"\{\{IMG:([^}]+)\}\}", img, t)
p = os.path.join(H, "curious-net-overview.html")
open(p, "w", encoding="utf-8").write(out)
print(p, round(len(out) / 1e6, 2), "MB", "placeholders left:", out.count("{{IMG"))
# print version for the PDF: every deep dive expanded (a PDF cannot be clicked open)
open(os.path.join(H, "_print.html"), "w", encoding="utf-8").write(out.replace('<details class="dive">', '<details class="dive" open>'))
