"""A small Markdown to HTML converter for the research notes the site shows, standard library only.

It covers what those notes use: headings, paragraphs, nested bullet and numbered lists, blockquotes, pipe tables
with alignment, bold, inline code, links and images. Three kinds of table get their own markup: a table whose body
cells are only images becomes a row of figures, a table whose first header cell reads "trained in" becomes the
transfer matrix, and every other table scrolls sideways inside its own box. The colour emoji of the transfer note
become swatches in the exact colours of its maps.
"""
from __future__ import annotations

import html
import re
import struct
from pathlib import Path

# The colour markers of transfer_setup.md: emoji -> (colour of the map, shape, what it stands for)
SWATCHES = {
    "🟩": ("#3cb44b", "sq", "agro-climate: temperate oceanic"),
    "🟦": ("#4363d8", "sq", "agro-climate: temperate sub-continental"),
    "🟨": ("#ffe119", "sq", "agro-climate: subtropics with winter rain (Mediterranean)"),
    "🔵": ("#4363d8", "ci", "AEZ: temperate cool, wet"),
    "🟢": ("#3cb44b", "ci", "AEZ: temperate cool, moist"),
    "🟡": ("#ffe119", "ci", "AEZ: temperate moderate"),
    "🟠": ("#f58231", "ci", "AEZ: subtropics moderately cool"),
    "🔴": ("#e6194b", "ci", "AEZ: subtropics cool"),
    "⚪": ("#ffffff", "ci", "AEZ: special class"),
}
LIST = re.compile(r"^( *)([-*]|\d+\.) +(.*)$")
HEADING = re.compile(r"^(#{1,6}) +(.*?) *#*$")
TABLE_SEP = re.compile(r"^\|? *:?-+:? *(\| *:?-+:? *)*\|? *$")


def slug(text: str) -> str:
    """GitHub's heading anchor: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"<[^>]+>", "", text).lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.strip().replace(" ", "-")


def png_size(path: Path) -> tuple[int, int] | None:
    """Width and height from a PNG header, so the page keeps its layout while images load."""
    if path.suffix.lower() != ".png" or not path.is_file():
        return None
    head = path.read_bytes()[:24]
    return struct.unpack(">II", head[16:24]) if head[:8] == b"\x89PNG\r\n\x1a\n" else None


class Converter:
    def __init__(self, asset_prefix: str, blob_prefix: str, id_prefix: str = "", heading_shift: int = 0,
                 source_dir: Path | None = None):
        self.asset_prefix = asset_prefix      # prepended to relative image paths
        self.source_dir = source_dir          # where those paths resolve on disk, to give images their size
        self.blob_prefix = blob_prefix        # where relative links to other files point
        self.id_prefix = id_prefix
        self.shift = heading_shift

    # -- inline ---------------------------------------------------------------------------------------------------
    def inline(self, text: str) -> str:
        codes: list[str] = []

        def keep(m):
            codes.append(f"<code>{html.escape(m.group(1))}</code>")
            return f"\x00{len(codes) - 1}\x00"

        text = re.sub(r"`([^`]+)`", keep, text)
        text = html.escape(text, quote=False).replace("&lt;br&gt;", "<br>")
        text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", lambda m: self.image(m.group(1), m.group(2)), text)
        text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: self.link(m.group(1), m.group(2)), text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        for emoji, (colour, shape, label) in SWATCHES.items():
            text = text.replace(emoji, f'<span class="sw {shape}" style="--c:{colour}" role="img" '
                                       f'aria-label="{html.escape(label)}" title="{html.escape(label)}"></span>')
        return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], text)

    def image(self, alt: str, src: str) -> str:
        size = ""
        if not re.match(r"^[a-z]+:|^/", src):
            if self.source_dir and (dims := png_size(self.source_dir / src)):
                size = f' width="{dims[0]}" height="{dims[1]}"'
            src = self.asset_prefix + src
        return (f'<a class="md-img" href="{html.escape(src)}" target="_blank" rel="noopener">'
                f'<img src="{html.escape(src)}" alt="{alt}"{size} loading="lazy"></a>')

    def link(self, label: str, href: str) -> str:
        if href.startswith("#"):
            return f'<a href="#{self.id_prefix}{href[1:]}">{label}</a>'
        if not re.match(r"^[a-z]+:|^/", href):
            return f'<a href="{html.escape(self.blob_prefix + href)}" target="_blank" rel="noopener">{label} ↗</a>'
        return f'<a href="{html.escape(href)}" target="_blank" rel="noopener">{label}</a>'

    # -- blocks ---------------------------------------------------------------------------------------------------
    def blocks(self, lines: list[str]) -> list[tuple[str, str]]:
        out, i, n = [], 0, len(lines)
        while i < n:
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            if m := HEADING.match(line):
                level = min(6, len(m.group(1)) + self.shift)
                text = m.group(2)
                out.append(("h", f'<h{level} id="{self.id_prefix}{slug(text)}" class="md-h{len(m.group(1))}">'
                                 f'{self.inline(text)}</h{level}>'))
                i += 1
            elif line.lstrip().startswith("|") and i + 1 < n and TABLE_SEP.match(lines[i + 1].strip()):
                j = i + 2
                while j < n and lines[j].lstrip().startswith("|"):
                    j += 1
                out.append(("table", self.table(lines[i], lines[i + 1], lines[i + 2:j])))
                i = j
            elif line.lstrip().startswith(">"):
                j = i
                while j < n and lines[j].lstrip().startswith(">"):
                    j += 1
                inner = [re.sub(r"^ *> ?", "", l) for l in lines[i:j]]
                out.append(("quote", "<blockquote>" + self.render(inner) + "</blockquote>"))
                i = j
            elif LIST.match(line):
                html_, i = self.list_block(lines, i)
                out.append(("list", html_))
            else:
                j = i + 1
                para = [line.strip()]
                while j < n and lines[j].strip() and not (HEADING.match(lines[j]) or LIST.match(lines[j])
                                                          or lines[j].lstrip().startswith((">", "|"))):
                    para.append(lines[j].strip())
                    j += 1
                text = " ".join(para)
                if re.fullmatch(r"!\[[^\]]*\]\([^)\s]+\)", text):
                    out.append(("fig", f'<figure class="md-fig">{self.inline(text)}</figure>'))
                else:
                    out.append(("p", f"<p>{self.inline(text)}</p>"))
                i = j
        return out

    def render(self, lines: list[str]) -> str:
        return "\n".join(h for _, h in self.blocks(lines))

    def list_block(self, lines: list[str], i: int) -> tuple[str, int]:
        first = LIST.match(lines[i])
        indent, ordered = len(first.group(1)), first.group(2)[0].isdigit()
        items, n = [], len(lines)
        while i < n:
            m = LIST.match(lines[i])
            if not m or len(m.group(1)) != indent or m.group(2)[0].isdigit() != ordered:
                break
            col = indent + len(m.group(2)) + 1
            content = [m.group(3)]
            i += 1
            while i < n:
                l = lines[i]
                if not l.strip():
                    k = i
                    while k < n and not lines[k].strip():
                        k += 1
                    if k < n and len(lines[k]) - len(lines[k].lstrip()) >= col:
                        content.extend([""] * (k - i))
                        i = k
                        continue
                    break
                lead = len(l) - len(l.lstrip())
                if lead <= indent and LIST.match(l):
                    break
                if lead < col and not content[-1].strip():
                    break
                content.append(l[min(lead, col):])
                i += 1
            blocks = self.blocks(content)
            body = "".join((h[3:-4] if k == 0 and kind == "p" else h) for k, (kind, h) in enumerate(blocks))
            items.append(f"<li>{body}</li>")
            k = i
            while k < n and not lines[k].strip():
                k += 1
            nxt = LIST.match(lines[k]) if k < n else None
            if not nxt or len(nxt.group(1)) != indent or nxt.group(2)[0].isdigit() != ordered:
                break
            i = k
        tag = "ol" if ordered else "ul"
        return f"<{tag}>" + "".join(items) + f"</{tag}>", i

    # -- tables ---------------------------------------------------------------------------------------------------
    @staticmethod
    def cells(row: str) -> list[str]:
        row = row.strip()
        row = row[1:] if row.startswith("|") else row
        row = row[:-1] if row.endswith("|") else row
        return [c.strip() for c in row.split("|")]

    def table(self, head: str, sep: str, rows: list[str]) -> str:
        header = self.cells(head)
        aligns = ["center" if c.startswith(":") and c.endswith(":") else "right" if c.endswith(":") else ""
                  for c in self.cells(sep)]
        body = [self.cells(r) for r in rows]
        if body and all(re.fullmatch(r"!\[[^\]]*\]\([^)]+\)", c) for r in body for c in r if c):
            figs = "".join(f'<figure>{self.inline(c)}<figcaption>{self.inline(h)}</figcaption></figure>'
                           for r in body for h, c in zip(header, r))
            return f'<div class="md-figs">{figs}</div>'
        matrix = "trained in" in header[0]

        def cell(tag, text, k, row=None):
            cls = []
            if aligns[k] if k < len(aligns) else "":
                cls.append(f"a-{aligns[k]}")
            if not matrix and len(re.sub(r"[*`]|!?\[[^\]]*\]\([^)]*\)", "", text)) <= 12:
                cls.append("nw")
            if matrix and tag == "td" and text.strip() == "◆":
                cls.append("diag")
            attr = f' class="{" ".join(cls)}"' if cls else ""
            if matrix and tag == "td" and k == 0:
                tag = "th"
                attr += ' scope="row"'
            return f"<{tag}{attr}>{self.inline(text)}</{tag}>"

        thead = "<tr>" + "".join(cell("th", h, k) for k, h in enumerate(header)) + "</tr>"
        tbody = "".join("<tr>" + "".join(cell("td", c, k) for k, c in enumerate(r)) + "</tr>" for r in body)
        cls = "md-table md-matrix" if matrix else "md-table"
        return f'<div class="{cls}" tabindex="0"><table><thead>{thead}</thead><tbody>{tbody}</tbody></table></div>'


def convert(text: str, **kw) -> str:
    return Converter(**kw).render(text.splitlines())
