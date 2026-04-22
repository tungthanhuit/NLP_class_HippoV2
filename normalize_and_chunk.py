#!/usr/bin/env python3
"""
normalize_and_chunk.py — Normalize + chunk văn bản pháp lý (Nghị định 168/2024/NĐ-CP)
                          thành corpus phù hợp cho multi-hop QA.

Usage:
    python normalize_and_chunk.py -i 168.txt -o ./out

Outputs (in --output-dir):
    chunks.jsonl     — 1 chunk/line với đầy đủ metadata (chuong, muc, dieu, khoan, diem)
    corpus.json      — format: [{"idx", "title", "text"}, ...]
    stats.txt        — summary thống kê

Chunking strategy (3-tier):
    Tier 1 — by_khoan:         1 khoản vừa vặn → 1 chunk
    Tier 2 — by_diem:          khoản dài → tách theo điểm (a, b, c, …)
    Tier 3 — by_*_split:       điểm vẫn dài → sentence-split có overlap

Mọi chunk đều chứa context header `Chương X > Mục Y > Điều N (title) > Khoản K > Điểm L`
in-line trong text → nhằm đảm bảo embedder không mất thông tin hierarchy.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# =============================================================================
# CONFIG
# =============================================================================
DEFAULT_MAX_CHARS = 1500  # ~400–500 tokens tiếng Việt; an toàn với BGE-M3 / multilingual-e5
DEFAULT_MIN_CHARS = 40    # drop micro-chunks (chỉ có header)
DEFAULT_OVERLAP = 120     # char-level overlap khi buộc phải hard-split


# =============================================================================
# 0. DATA MODEL
# =============================================================================
@dataclass
class Diem:
    letter: str   # 'a', 'b', 'c', ..., 'đ'
    text: str


@dataclass
class Khoan:
    number: int
    intro: str                              # text sau "N." và trước điểm đầu tiên
    diems: list[Diem] = field(default_factory=list)

    def full_text(self) -> str:
        parts = []
        if self.intro.strip():
            parts.append(self.intro.strip())
        for d in self.diems:
            parts.append(f"{d.letter}) {d.text.strip()}")
        return "\n".join(parts)


@dataclass
class Dieu:
    number: int
    title: str
    chuong: Optional[str]
    chuong_title: str
    muc: Optional[str]
    muc_title: Optional[str]
    khoans: list[Khoan] = field(default_factory=list)


# =============================================================================
# 1. NORMALIZATION
# =============================================================================
def normalize_text(raw: str) -> str:
    """NFC unicode, chuẩn hóa line endings, gộp whitespace thừa, normalize quotes."""
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Smart quotes → ASCII (đỡ rắc rối cho regex và embedder)
    text = (text
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2018", "'").replace("\u2019", "'"))
    # Gộp tab/nhiều space thành 1 space
    text = re.sub(r"[ \t]+", " ", text)
    # Giới hạn blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def resolve_references(dieus: list[Dieu], decree_name: str = "Nghị định 168/2024/NĐ-CP") -> dict:
    """
    Resolve TẤT CẢ các tham chiếu "này" (in-place) về entity cụ thể.
    ĐÂY LÀ BƯỚC BẮT BUỘC cho HippoRAG: nếu không resolve, LLM sẽ extract ra các triples
    ambiguous như (X, quy_định_tại, "khoản này") — vô nghĩa khi đứng ngoài context gốc.

    Xử lý các loại sau:
      - "Điều này"       → "Điều N"            (N = số điều hiện tại)
      - "khoản này"      → "khoản K"           (K = số khoản hiện tại)
      - "điểm này"       → "điểm L"            (L = chữ điểm hiện tại)
      - "Nghị định này"  → decree_name         (mặc định NĐ 168/2024/NĐ-CP)

    Lưu ý về "khoản này":
      - Trong khoan.intro  → trỏ về chính khoản đó
      - Trong diem.text    → trỏ về khoản cha của điểm (cùng giá trị)

    Trả về: dict đếm số lần thay thế cho từng loại (để log/audit).
    """
    # Regex cho các pattern — case-insensitive, gộp whitespace linh hoạt
    RE_DIEU_NAY     = re.compile(r"Điều\s+này", re.IGNORECASE)
    RE_KHOAN_NAY    = re.compile(r"khoản\s+này", re.IGNORECASE)
    RE_DIEM_NAY     = re.compile(r"điểm\s+này", re.IGNORECASE)
    RE_NGHI_DINH_NAY = re.compile(r"Nghị\s+định\s+này", re.IGNORECASE)

    stats = {"Điều này": 0, "khoản này": 0, "điểm này": 0, "Nghị định này": 0}

    def replace_in(text: str, dieu_num: int, khoan_num: Optional[int],
                   diem_letter: Optional[str]) -> str:
        nonlocal stats
        # "Nghị định này" — thay thế đầu tiên vì đơn giản nhất, không phụ thuộc context
        text, n = RE_NGHI_DINH_NAY.subn(decree_name, text)
        stats["Nghị định này"] += n

        # "Điều này" → "Điều N"
        text, n = RE_DIEU_NAY.subn(f"Điều {dieu_num}", text)
        stats["Điều này"] += n

        # "khoản này" → "khoản K" (chỉ khi biết khoản hiện tại)
        if khoan_num is not None:
            text, n = RE_KHOAN_NAY.subn(f"khoản {khoan_num}", text)
            stats["khoản này"] += n

        # "điểm này" → "điểm L" (chỉ khi biết điểm hiện tại)
        if diem_letter is not None:
            text, n = RE_DIEM_NAY.subn(f"điểm {diem_letter}", text)
            stats["điểm này"] += n

        return text

    for dieu in dieus:
        for khoan in dieu.khoans:
            # Trong khoan.intro: không có điểm hiện tại → diem_letter=None
            khoan.intro = replace_in(khoan.intro, dieu.number, khoan.number, None)
            for diem in khoan.diems:
                # Trong diem.text: cả khoản cha và điểm hiện tại đều có
                diem.text = replace_in(diem.text, dieu.number, khoan.number, diem.letter)

    return stats


# Giữ backward-compatible alias để không break code cũ
def resolve_dieu_nay_references(dieus: list[Dieu]) -> int:
    """[DEPRECATED] Sử dụng resolve_references() để xử lý đầy đủ các pattern 'này'."""
    stats = resolve_references(dieus)
    return sum(stats.values())


# =============================================================================
# 2. PARSE HIERARCHY (state machine)
# =============================================================================
# Regex cho từng cấp structural marker
RE_CHUONG = re.compile(r"^Chương\s+([IVXLCDM]+)\s*$")
RE_MUC    = re.compile(r"^Mục\s+(\d+)\.\s*(.+)$", re.DOTALL)
RE_DIEU   = re.compile(r"^Điều\s+(\d+)\.\s*(.+)$", re.DOTALL)
RE_KHOAN  = re.compile(r"^(\d+)\.\s+(.+)$", re.DOTALL)
RE_DIEM   = re.compile(r"^([a-zđ])\)\s+(.+)$", re.DOTALL)


def parse_document(text: str) -> list[Dieu]:
    """
    Parse văn bản thành list[Dieu], mỗi Dieu có list[Khoan], mỗi Khoan có list[Diem].
    State machine chạy qua từng paragraph (tách bởi blank line).
    """
    paragraphs = re.split(r"\n\s*\n", text)

    dieus: list[Dieu] = []
    current_chuong: Optional[str] = None
    current_chuong_title: str = ""
    current_muc: Optional[str] = None
    current_muc_title: Optional[str] = None
    current_dieu: Optional[Dieu] = None
    current_khoan: Optional[Khoan] = None
    expecting_chuong_title = False

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Chương title paragraph (đi ngay sau "Chương N")
        if expecting_chuong_title:
            current_chuong_title = para
            expecting_chuong_title = False
            continue

        if m := RE_CHUONG.match(para):
            current_chuong = m.group(1)
            current_muc = None
            current_muc_title = None
            expecting_chuong_title = True
            continue

        if m := RE_MUC.match(para):
            current_muc = m.group(1)
            current_muc_title = m.group(2).strip()
            continue

        if m := RE_DIEU.match(para):
            current_dieu = Dieu(
                number=int(m.group(1)),
                title=m.group(2).strip(),
                chuong=current_chuong,
                chuong_title=current_chuong_title,
                muc=current_muc,
                muc_title=current_muc_title,
            )
            dieus.append(current_dieu)
            current_khoan = None
            continue

        # Khoản & điểm chỉ hợp lệ khi đang ở trong 1 Điều
        if current_dieu is not None and (m := RE_KHOAN.match(para)):
            current_khoan = Khoan(
                number=int(m.group(1)),
                intro=m.group(2).strip(),
            )
            current_dieu.khoans.append(current_khoan)
            continue

        if current_khoan is not None and (m := RE_DIEM.match(para)):
            current_khoan.diems.append(Diem(letter=m.group(1), text=m.group(2).strip()))
            continue

        # Continuation: paragraph không match cấu trúc nào → append vào leaf node gần nhất
        if current_khoan is not None:
            if current_khoan.diems:
                current_khoan.diems[-1].text += "\n" + para
            else:
                current_khoan.intro += "\n" + para
        # (Các paragraph trước Điều đầu tiên bị bỏ qua — thường là header văn bản)

    return dieus


# =============================================================================
# 3. CHUNKING (3-tier)
# =============================================================================
def make_context_header(
    dieu: Dieu,
    khoan_num: Optional[int] = None,
    diem_letter: Optional[str] = None,
    title_max: int = 80,
) -> str:
    """Header gọn, đi kèm mọi chunk, đảm bảo embedder giữ được hierarchy."""
    title = dieu.title if len(dieu.title) <= title_max else dieu.title[:title_max] + "..."
    parts = []
    if dieu.chuong:
        parts.append(f"Chương {dieu.chuong}")
    if dieu.muc:
        parts.append(f"Mục {dieu.muc}")
    parts.append(f"Điều {dieu.number} ({title})")
    if khoan_num is not None:
        parts.append(f"Khoản {khoan_num}")
    if diem_letter is not None:
        parts.append(f"Điểm {diem_letter}")
    return " > ".join(parts)


def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Fallback Tier-3: sentence-aware split, hard-split nếu câu đơn > max_chars.
    Ưu tiên split tại ranh giới câu (. ; ? !) để giảm thiểu mất ngữ nghĩa.
    """
    if len(text) <= max_chars:
        return [text]

    # Split theo ranh giới câu (giữ dấu câu ở cuối câu)
    sentences = re.split(r"(?<=[.;?!])\s+", text)
    chunks: list[str] = []
    current = ""

    for s in sentences:
        if not s:
            continue
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip() if current else s
        else:
            if current:
                chunks.append(current)
                current = ""
            if len(s) <= max_chars:
                current = s
            else:
                # Câu quá dài, hard-split với overlap
                step = max(1, max_chars - overlap)
                for i in range(0, len(s), step):
                    chunks.append(s[i:i + max_chars])
    if current:
        chunks.append(current)

    return chunks


def chunk_dieus(
    dieus: list[Dieu],
    max_chars: int,
    min_chars: int,
    overlap: int,
) -> list[dict]:
    """
    Áp dụng 3-tier strategy:
      Tier 1 (by_khoan)        : khoản.full_text() + header <= max_chars
      Tier 2 (by_diem)         : khoản dài → 1 chunk cho intro + 1 chunk cho mỗi điểm
      Tier 3 (by_*_split)      : điểm hoặc khoản-không-điểm vẫn dài → sentence split
    """
    chunks: list[dict] = []

    def make_chunk(method: str, dieu: Dieu, khoan: Optional[Khoan],
                   diem_letter: Optional[str], text: str,
                   header_override: Optional[str] = None) -> dict:
        khoan_num = khoan.number if khoan else None
        header = header_override or make_context_header(dieu, khoan_num, diem_letter)
        return {
            "chunk_id": "",  # fill sau
            "chunk_method": method,
            "chuong": dieu.chuong,
            "chuong_title": dieu.chuong_title,
            "muc": dieu.muc,
            "muc_title": dieu.muc_title,
            "dieu": dieu.number,
            "dieu_title": dieu.title,
            "khoan": khoan_num,
            "diem": diem_letter,
            "context_header": header,
            "text": text,
            "char_length": len(text),
        }

    for dieu in dieus:
        # Edge case: Điều chỉ có title, không có khoản (e.g. Điều 55)
        if not dieu.khoans:
            header = make_context_header(dieu)
            text = f"[{header}]\n{dieu.title}"
            chunks.append(make_chunk("dieu_only", dieu, None, None, text))
            continue

        for khoan in dieu.khoans:
            full_text = khoan.full_text()
            header_khoan = make_context_header(dieu, khoan.number)
            body_prefix = f"[{header_khoan}]\n{dieu.title}\n\n"
            full_chunk_text = body_prefix + full_text

            # --- Tier 1: khoản vừa vặn ---
            if len(full_chunk_text) <= max_chars:
                chunks.append(make_chunk("by_khoan", dieu, khoan, None, full_chunk_text))
                continue

            # --- Tier 2: khoản dài, có điểm → tách theo điểm ---
            if khoan.diems:
                # Intro của khoản (nếu có nội dung đáng kể) → 1 chunk riêng
                if khoan.intro.strip():
                    intro_text = f"[{header_khoan} (intro)]\n{dieu.title}\n\n{khoan.intro.strip()}"
                    # Intro cũng có thể rất dài → Tier-3 nếu cần
                    if len(intro_text) <= max_chars:
                        chunks.append(make_chunk("by_khoan_intro", dieu, khoan, None, intro_text))
                    else:
                        body_budget = max_chars - len(header_khoan) - 120
                        sub = split_long_text(khoan.intro.strip(), body_budget, overlap)
                        for i, s in enumerate(sub):
                            h = f"{header_khoan} (intro, phần {i+1}/{len(sub)})"
                            t = f"[{h}]\n{dieu.title}\n\n{s}"
                            chunks.append(make_chunk("by_khoan_intro_split", dieu, khoan, None, t,
                                                    header_override=h))

                # Mỗi điểm → chunk riêng
                # Lấy câu đầu của intro để đưa vào prefix mỗi điểm (giữ ngữ cảnh mức phạt)
                intro_first_line = khoan.intro.strip().split("\n")[0][:180]

                for diem in khoan.diems:
                    header_diem = make_context_header(dieu, khoan.number, diem.letter)
                    ctx_line = (f"[Khoản {khoan.number} nói chung: {intro_first_line}]"
                                if intro_first_line else "")
                    diem_body = f"{diem.letter}) {diem.text.strip()}"
                    prefix = f"[{header_diem}]\n{dieu.title}\n{ctx_line}\n\n"
                    full_diem_text = prefix + diem_body

                    # Tier 2 đủ
                    if len(full_diem_text) <= max_chars:
                        chunks.append(make_chunk("by_diem", dieu, khoan, diem.letter, full_diem_text))
                    else:
                        # Tier 3: điểm quá dài → sentence split
                        body_budget = max_chars - len(prefix) - 50
                        sub = split_long_text(diem_body, body_budget, overlap)
                        for i, s in enumerate(sub):
                            h = f"{header_diem} (phần {i+1}/{len(sub)})"
                            t = f"[{h}]\n{dieu.title}\n{ctx_line}\n\n{s}"
                            chunks.append(make_chunk("by_diem_split", dieu, khoan,
                                                    f"{diem.letter}.{i+1}", t,
                                                    header_override=h))
            else:
                # --- Tier 3: khoản dài, không có điểm → sentence split trực tiếp ---
                body_budget = max_chars - len(body_prefix) - 50
                sub = split_long_text(full_text, body_budget, overlap)
                for i, s in enumerate(sub):
                    h = f"{header_khoan} (phần {i+1}/{len(sub)})"
                    t = f"[{h}]\n{dieu.title}\n\n{s}"
                    chunks.append(make_chunk("by_khoan_split", dieu, khoan, None, t,
                                            header_override=h))

    # Drop micro-chunks (chỉ còn header, không có content thực)
    chunks = [c for c in chunks if c["char_length"] >= min_chars]

    # Re-index chunk_id
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"c{i+1:04d}"

    return chunks


# =============================================================================
# 4. OUTPUTS
# =============================================================================
def save_jsonl(chunks: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def save_hipporag_corpus(chunks: list[dict], path: Path) -> None:
    """
    Format HippoRAG-compatible: list of dicts với {idx, title, text}.
    HippoRAG v2 chấp nhận format này; title dùng làm passage identifier.
    """
    corpus = [
        {"idx": i, "title": c["context_header"], "text": c["text"]}
        for i, c in enumerate(chunks)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)


def compute_stats(chunks: list[dict]) -> dict:
    lengths = [c["char_length"] for c in chunks]
    methods = Counter(c["chunk_method"] for c in chunks)
    sorted_len = sorted(lengths)
    n = len(lengths)

    return {
        "count": n,
        "min": min(lengths),
        "max": max(lengths),
        "mean": sum(lengths) / n,
        "median": sorted_len[n // 2],
        "p25": sorted_len[n // 4],
        "p75": sorted_len[3 * n // 4],
        "p95": sorted_len[int(n * 0.95)],
        "p99": sorted_len[int(n * 0.99)] if n >= 100 else sorted_len[-1],
        "method_counts": dict(methods),
    }


def write_stats_file(stats: dict, chunks: list[dict], path: Path) -> str:
    per_dieu = Counter(c["dieu"] for c in chunks)
    per_chuong = Counter(c["chuong"] for c in chunks)

    lines = [
        "=" * 60,
        "CHUNKING SUMMARY",
        "=" * 60,
        f"Total chunks     : {stats['count']}",
        "",
        "Length stats (chars):",
        f"  min    = {stats['min']}",
        f"  P25    = {stats['p25']}",
        f"  median = {stats['median']}",
        f"  mean   = {stats['mean']:.1f}",
        f"  P75    = {stats['p75']}",
        f"  P95    = {stats['p95']}",
        f"  P99    = {stats['p99']}",
        f"  max    = {stats['max']}",
        "",
        "Method breakdown:",
    ]
    for m, c in sorted(stats["method_counts"].items(), key=lambda x: -x[1]):
        pct = 100 * c / stats["count"]
        lines.append(f"  {m:<25s}  {c:>4d}  ({pct:5.1f}%)")

    lines.append("")
    lines.append("Chunks per Chương:")
    for ch, c in sorted(per_chuong.items(), key=lambda x: (x[0] or "")):
        lines.append(f"  Chương {ch}: {c}")

    lines.append("")
    lines.append("Top 10 Điều nhiều chunk nhất (hub candidates):")
    for d, c in per_dieu.most_common(10):
        lines.append(f"  Điều {d}: {c} chunks")

    summary = "\n".join(lines)
    path.write_text(summary, encoding="utf-8")
    return summary


# =============================================================================
# 5. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", "-i", required=True, help="Path to raw text file")
    ap.add_argument("--output-dir", "-o", default="./out", help="Output directory")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help=f"Max chars per chunk (default {DEFAULT_MAX_CHARS})")
    ap.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS,
                    help=f"Drop chunks shorter than this (default {DEFAULT_MIN_CHARS})")
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                    help=f"Overlap for hard-split (default {DEFAULT_OVERLAP})")
    ap.add_argument("--no-resolve-dieu-nay", action="store_true",
                    help="Disable 'Điều này' → 'Điều N' resolution")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Reading {input_path}")
    raw = input_path.read_text(encoding="utf-8")
    print(f"      size: {len(raw):,} chars")

    print(f"[2/5] Normalizing")
    text = normalize_text(raw)
    print(f"      after NFC + cleanup: {len(text):,} chars")

    print(f"[3/5] Parsing structure")
    dieus = parse_document(text)
    n_khoan = sum(len(d.khoans) for d in dieus)
    n_diem = sum(len(k.diems) for d in dieus for k in d.khoans)
    print(f"      parsed: {len(dieus)} điều | {n_khoan} khoản | {n_diem} điểm")

    if not args.no_resolve_dieu_nay:
        stats = resolve_references(dieus)
        total = sum(stats.values())
        print(f"      resolved {total} references:")
        for name, n in stats.items():
            if n > 0:
                print(f"        - '{name}' → cụ thể: {n} lần")

    print(f"[4/5] Chunking (max_chars={args.max_chars}, overlap={args.overlap})")
    chunks = chunk_dieus(dieus, args.max_chars, args.min_chars, args.overlap)
    print(f"      produced {len(chunks)} chunks")

    print(f"[5/5] Writing outputs → {out_dir.resolve()}")
    save_jsonl(chunks, out_dir / "chunks.jsonl")
    save_hipporag_corpus(chunks, out_dir / "corpus.json")
    stats = compute_stats(chunks)
    summary = write_stats_file(stats, chunks, out_dir / "stats.txt")

    print()
    print(summary)
    print()
    print("Output files:")
    for fn in ["chunks.jsonl", "corpus.json", "histogram.png", "stats.txt"]:
        fp = out_dir / fn
        if fp.exists():
            print(f"  - {fp} ({fp.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
