"""Plain-text extraction from the three resume formats we support.

    .pdf  -> pdfplumber
    .tex  -> the LaTeX stripper below
    .txt  -> read as-is

Nothing in here talks to the network. ``pdfplumber`` is imported lazily so that
a machine without it can still process .tex/.txt files.
"""

from __future__ import annotations

import re
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".tex", ".txt"}

# pdfplumber emits "(cid:136)" when a glyph has no usable ToUnicode mapping.
# It is almost always a decorative bullet, and it is pure noise downstream.
_CID_RE = re.compile(r"\(cid:\d+\)")


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def extract_text(path: Path) -> str:
    """Return plain text for one resume file.

    Raises ExtractionError on anything we cannot read, so the caller can skip
    the file and keep going rather than aborting the whole build.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _extract_pdf(path)
        if suffix == ".tex":
            return strip_latex(_read_text(path))
        if suffix == ".txt":
            return _read_text(path)
    except ExtractionError:
        raise
    except Exception as exc:  # pdfplumber raises a wide variety of errors
        raise ExtractionError(f"{path.name}: {type(exc).__name__}: {exc}") from exc
    raise ExtractionError(f"{path.name}: unsupported extension {suffix!r}")


class ExtractionError(RuntimeError):
    """Raised when a single resume file cannot be read."""


def _read_text(path: Path) -> str:
    # Resumes exported from Word/Overleaf are occasionally latin-1 or have a BOM.
    return path.read_text(encoding="utf-8-sig", errors="replace")


def _extract_pdf(path: Path) -> str:
    import pdfplumber  # lazy: only needed when a .pdf is actually present

    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    text = _CID_RE.sub(" ", "\n".join(pages))
    if not text.strip():
        # A scanned/image-only PDF yields nothing. Say so instead of silently
        # contributing an empty document to the profile.
        raise ExtractionError(
            f"{path.name}: no extractable text (likely a scanned image; OCR required)"
        )
    return text


# --------------------------------------------------------------------------
# LaTeX stripping
# --------------------------------------------------------------------------
#
# Goal: keep the readable content, drop the markup. A full LaTeX parser is out
# of scope, but resume templates use a small, predictable subset: a preamble we
# can discard wholesale, a handful of text-formatting commands, itemize/tabular
# environments, and template-specific macros like \resumeSubheading.
#
# The strategy is a single left-to-right walk with a balanced-brace reader:
#   * known commands consume a fixed number of {...} groups, and we keep the
#     arguments that carry visible text;
#   * unknown commands keep their group contents (custom macros in resume
#     templates almost always wrap text);
#   * commands on the drop list consume their groups and emit nothing.

# name -> (number of required {...} groups, indices of the groups to keep)
_ARGS_KEEP: dict[str, tuple[int, list[int]]] = {
    "textbf": (1, [0]),
    "textit": (1, [0]),
    "textsc": (1, [0]),
    "texttt": (1, [0]),
    "textrm": (1, [0]),
    "emph": (1, [0]),
    "underline": (1, [0]),
    "uline": (1, [0]),
    "text": (1, [0]),
    "mbox": (1, [0]),
    "makebox": (1, [0]),
    "section": (1, [0]),
    "subsection": (1, [0]),
    "subsubsection": (1, [0]),
    "paragraph": (1, [0]),
    "title": (1, [0]),
    "author": (1, [0]),
    "href": (2, [1]),  # \href{url}{label} -> label
    "textcolor": (2, [1]),  # \textcolor{red}{text} -> text
    "colorbox": (2, [1]),
    "url": (1, [0]),
    "footnote": (1, [0]),
    # Common resume-template macros (Jake's / Deedy / sb2nov style).
    "resumeItem": (1, [0]),
    "resumeSubItem": (1, [0]),
    "resumeSubheading": (4, [0, 1, 2, 3]),
    "resumeProjectHeading": (2, [0, 1]),
    "resumeSubSubheading": (2, [0, 1]),
    "resumeEducationHeading": (6, [0, 1, 2, 3, 4, 5]),
    "cvitem": (2, [0, 1]),
    "cventry": (6, [0, 1, 2, 3, 4, 5]),
}

# Commands whose arguments are markup, not content: consume and emit nothing.
_DROP_WITH_ARGS: frozenset[str] = frozenset(
    {
        "documentclass", "usepackage", "newcommand", "renewcommand",
        "providecommand", "newenvironment", "renewenvironment", "definecolor",
        "setlength", "addtolength", "setcounter", "vspace", "hspace",
        "pagestyle", "thispagestyle", "titleformat", "titlespacing",
        "titlerule", "geometry", "hypersetup", "fontfamily", "fontsize",
        "selectfont", "input", "include", "label", "ref", "pageref", "cite",
        "includegraphics", "raisebox", "scalebox", "resizebox", "rule",
        "color", "pagenumbering", "bibliographystyle", "bibliography",
        "captionsetup", "usetikzlibrary", "tikzset", "graphicspath",
        "AtBeginDocument", "pdfgentounicode", "faIcon",
    }
)

# Zero-argument commands -> literal replacement.
_SIMPLE: dict[str, str] = {
    "item": "\n\u2022 ",
    "par": "\n\n",
    "newline": "\n",
    "linebreak": "\n",
    "hline": "\n",
    "hrule": "\n",
    "bigskip": "\n",
    "medskip": "\n",
    "smallskip": "\n",
    "newpage": "\n",
    "clearpage": "\n",
    "vfill": "\n",
    "hfill": " ",
    "quad": " ", "qquad": " ", "enspace": " ", "thinspace": " ",
    "noindent": "", "indent": "", "centering": "", "raggedright": "",
    "raggedleft": "", "left": "", "right": "", "sloppy": "",
    "tiny": "", "scriptsize": "", "footnotesize": "", "small": "",
    "normalsize": "", "large": "", "Large": "", "LARGE": "", "huge": "",
    "Huge": "", "bfseries": "", "itshape": "", "scshape": "", "rmfamily": "",
    "ttfamily": "", "sffamily": "", "bf": "", "it": "", "sc": "", "rm": "",
    "today": "", "maketitle": "", "vrule": "", "strut": "",
    "LaTeX": "LaTeX", "TeX": "TeX", "ldots": "...", "dots": "...",
    "textbullet": "\u2022", "textbar": "|", "textbackslash": "\\",
    "degree": "\u00b0", "&": "&",
}

# \& \% \_ ... -> the literal character.
_ESCAPED_CHARS = set("&%$#_{}~^")

# Environments whose *content* is markup we don't want at all.
_DROP_ENVIRONMENTS = ("tikzpicture", "picture", "comment", "verbatim*", "filecontents")

# \begin{tabular}{|l|r|} -- the second group is a column spec, not content.
_ENVS_WITH_SPEC = {"tabular", "tabular*", "tabularx", "array", "longtable", "supertabular"}


def strip_latex(src: str) -> str:
    """Convert LaTeX source to readable plain text."""
    src = _remove_comments(src)

    # Discard the preamble entirely if there is a document body.
    body = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", src, re.S)
    if body:
        src = body.group(1)

    for env in _DROP_ENVIRONMENTS:
        src = re.sub(
            r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}",
            "",
            src,
            flags=re.S,
        )

    return _cleanup(_expand(src))


def _remove_comments(src: str) -> str:
    """Strip `%` comments, respecting `\\%` escapes."""
    out_lines: list[str] = []
    for line in src.splitlines():
        buf: list[str] = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\" and i + 1 < len(line):
                buf.append(line[i : i + 2])
                i += 2
                continue
            if ch == "%":
                break
            buf.append(ch)
            i += 1
        out_lines.append("".join(buf))
    return "\n".join(out_lines)


def _read_group(src: str, i: int) -> tuple[int, str | None]:
    """Read a balanced `{...}` group starting at or after position `i`.

    Returns (index after the group, contents) or (i, None) if there is no group.
    """
    j = i
    while j < len(src) and src[j] in " \t\n\r":
        j += 1
    if j >= len(src) or src[j] != "{":
        return i, None
    depth = 0
    start = j + 1
    while j < len(src):
        ch = src[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return j + 1, src[start:j]
        j += 1
    return len(src), src[start:]  # unbalanced source; salvage the tail


def _skip_optional(src: str, i: int) -> int:
    """Skip any `[...]` optional arguments at position `i`."""
    while True:
        j = i
        while j < len(src) and src[j] in " \t":
            j += 1
        if j < len(src) and src[j] == "[":
            depth = 0
            while j < len(src):
                if src[j] == "[":
                    depth += 1
                elif src[j] == "]":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            i = j
            continue
        return i


def _expand(src: str) -> str:
    out: list[str] = []
    i = 0
    n = len(src)

    while i < n:
        ch = src[i]

        if ch == "\\":
            if i + 1 < n and src[i + 1] in _ESCAPED_CHARS:
                out.append(src[i + 1])
                i += 2
                continue
            if i + 1 < n and src[i + 1] == "\\":
                out.append("\n")
                i += 2
                # \\[12pt] -- drop the spacing argument
                i = _skip_optional(src, i)
                continue

            m = re.match(r"\\([A-Za-z@]+)(\*?)", src[i:])
            if not m:
                i += 1  # a lone backslash before punctuation; drop it
                continue
            name = m.group(1)
            i += len(m.group(0))

            if name in ("begin", "end"):
                i, env = _read_group(src, i)
                i = _skip_optional(src, i)
                if name == "begin" and (env or "").strip() in _ENVS_WITH_SPEC:
                    i, _ = _read_group(src, i)  # column spec -> discard
                out.append("\n")
                continue

            if name in _SIMPLE:
                out.append(_SIMPLE[name])
                i = _skip_optional(src, i)
                continue

            i = _skip_optional(src, i)

            if name in _DROP_WITH_ARGS:
                # \newcommand{\x}[2]{...} -- groups and optionals interleave.
                while True:
                    i = _skip_optional(src, i)
                    i, grp = _read_group(src, i)
                    if grp is None:
                        break
                continue

            if name in _ARGS_KEEP:
                count, keep = _ARGS_KEEP[name]
                groups: list[str] = []
                for _ in range(count):
                    i = _skip_optional(src, i)
                    i, grp = _read_group(src, i)
                    if grp is None:
                        break
                    groups.append(grp)
                kept = [_expand(groups[k]).strip() for k in keep if k < len(groups)]
                out.append(" \u2014 ".join(p for p in kept if p))
                continue

            # Unknown command: keep the contents of its groups. Custom resume
            # macros are nearly always thin wrappers around visible text.
            i, grp = _read_group(src, i)
            if grp is not None:
                out.append(_expand(grp))
            else:
                out.append(" ")
            continue

        if ch == "{" or ch == "}":
            i += 1  # bare grouping braces carry no content
            continue

        if ch == "$":
            # Inline math in a resume is almost always a stray symbol.
            i += 1
            continue

        if ch == "&":
            out.append(" | ")  # tabular column separator
            i += 1
            continue

        if ch == "~":
            out.append(" ")
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _cleanup(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CID_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\| *(\| *)+", " | ", text)  # collapse empty tabular cells
    text = "\n".join(line.strip().strip("|").strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
