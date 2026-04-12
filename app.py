import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from itertools import combinations
import io
import re

# ─────────────────────────────────────────────
#  Page config & custom CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BankRecon Pro",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Dark gradient background ── */
  .stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
  }
  [data-testid="stSidebar"] * { color: #e6edf3 !important; }

  /* ── Metric cards ── */
  [data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
  }

  /* ── Section headers ── */
  .section-header {
    background: linear-gradient(90deg, #1f6feb 0%, #388bfd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
  }

  /* ── Status badges ── */
  .badge-matched   { background:#1a4731; color:#3fb950; border:1px solid #3fb950;
                     padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
  .badge-mismatch  { background:#4a1a1a; color:#f85149; border:1px solid #f85149;
                     padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
  .badge-subset    { background:#3a2d00; color:#e3b341; border:1px solid #e3b341;
                     padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
  .badge-unmatched { background:#2d1a4a; color:#d2a8ff; border:1px solid #d2a8ff;
                     padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }

  /* ── Upload box ── */
  [data-testid="stFileUploader"] {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 12px;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid #30363d;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e !important;
    border-radius: 8px;
    font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: #1f6feb !important;
    color: #fff !important;
  }

  /* ── Highlighted rows ── */
  .highlight-red   { background-color: rgba(248, 81, 73, 0.12) !important; }
  .highlight-green { background-color: rgba(63, 185, 80, 0.10) !important; }
  .highlight-amber { background-color: rgba(227, 179, 65, 0.12) !important; }

  /* ── Divider ── */
  hr { border-color: #30363d; }

  /* ── Info boxes ── */
  .info-box {
    background: #161b22; border: 1px solid #30363d; border-radius:10px;
    padding: 16px; margin-bottom:12px;
  }

  /* ── Scrollable table container ── */
  .table-container { overflow-x: auto; border-radius: 8px; }

  /* ── Streamlit table override ── */
  .dataframe { background: #161b22 !important; color: #e6edf3 !important; }
  thead tr th { background: #21262d !important; color: #58a6ff !important; }
  tbody tr:hover { background: #1c2128 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helper utilities
# ─────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip punctuation/extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# ── Column keyword maps — comprehensive Indian & global bank terminology ────────
BANK_COL_MAP = {
    "Bank Ref": [
        "ref", "txn", "transaction", "chequeno", "chqno", "chq",
        "cheque", "check", "instrument", "serial", "slno",
        "utr", "rrn", "neft", "rtgs", "imps", "ifsc",
        "chqrefno", "refno", "transactionid", "txnid",
    ],
    "Bank Date": [
        "date", "dt", "posting", "valuedate", "transactiondate",
        "txndate", "trandate", "valuedt", "bookingdate",
        "settlementdate", "effectivedate",
    ],
    "Bank Amount": [
        "amount", "amt", "debit", "credit", "withdrawal",
        "deposit", "sum", "net", "gross",
        "debitamt", "creditamt", "withdrawalamt", "depositamt",
        "dramt", "cramt", "debited", "credited",
        "withdrawalamtdr", "depositamtcr",
        "transactionamount", "txnamount",
    ],
    "Bank Description": [
        "description", "desc", "narration", "narrative",
        "particulars", "details", "memo", "remarks", "text",
        "transactionremarks", "txnremarks", "transactionparticulars",
        "transactiondescription", "paymentremarks", "transferremarks",
        "remarks", "additionaltransactioninformation",
    ],
}

GL_COL_MAP = {
    "GL Vendor Name": [
        "vendor", "supplier", "party", "name", "payee", "customer",
        "partyname", "vendorname", "suppliername", "ledger", "account",
        "accountname", "counterparty",
    ],
    "GL Amount": [
        "amount", "amt", "debit", "credit", "value", "sum",
        "amountinlocalcurrency", "baseamount", "localamount", "foreignamount",
        "invoiceamount", "paymentamount", "transactionamount",
    ],
    "GL Date": [
        "date", "dt", "posting", "postingdate", "documentdate",
        "entrydate", "invoicedate", "valuedate", "duedate",
    ],
    "GL Reference": [
        "ref", "reference", "voucher", "invoice", "id", "number", "no",
        "voucherno", "invoiceno", "documentno", "docno", "journalno",
        "assignment", "documentnumber",
    ],
}


def _col_key(col: str) -> str:
    """Normalise a column header: lowercase, strip all non-alphanumeric."""
    return re.sub(r'[^a-z0-9]', '', str(col).lower())


def _looks_numeric(series: pd.Series, sample: int = 20) -> bool:
    """Return True if >60% of non-null values in a sample look like numbers."""
    vals = series.dropna().head(sample)
    if len(vals) == 0:
        return False
    hits = 0
    for v in vals:
        try:
            float(str(v).replace(',', '').replace('(', '-').replace(')', ''))
            hits += 1
        except ValueError:
            pass
    return hits / len(vals) > 0.6


def _looks_like_date(series: pd.Series, sample: int = 20) -> bool:
    """Return True if >60% of non-null values look like dates."""
    import re as _re
    date_pat = _re.compile(
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        r'|\d{4}[/-]\d{1,2}[/-]\d{1,2}'
        r'|\d{1,2}\s+\w{3}\s+\d{2,4}'
    )
    vals = series.dropna().head(sample)
    if len(vals) == 0:
        return False
    hits = sum(1 for v in vals if date_pat.search(str(v)))
    return hits / len(vals) > 0.6


def _looks_like_text(series: pd.Series, sample: int = 20) -> bool:
    """Return True if most values are non-numeric strings with spaces (descriptions)."""
    vals = series.dropna().head(sample)
    if len(vals) == 0:
        return False
    hits = 0
    for v in vals:
        s = str(v).strip()
        try:
            float(s.replace(',', ''))
        except ValueError:
            if len(s) > 5 and ' ' in s:
                hits += 1
    return hits / len(vals) > 0.4


def auto_map_columns(df: pd.DataFrame,
                     col_map: dict,
                     required: list) -> tuple[dict, list]:
    """
    Detect mapping from col_map → df.columns.
    Returns:
      - detected mapping {canonical: original}
      - list of missing required canonical names
    """
    remaining_cols = list(df.columns)
    detected   = {}
    missing    = []

    for canonical, keywords in col_map.items():
        ck  = _col_key(canonical)
        matched = None

        # ── Pass 1: exact ─────────────────────────────────────────────
        for col in remaining_cols:
            if _col_key(col) == ck:
                matched = col; break

        # ── Pass 2: substring of canonical key ────────────────────────
        if not matched:
            for col in remaining_cols:
                cn = _col_key(col)
                if (ck in cn and len(ck) >= 4) or (cn in ck and len(cn) >= 4):
                    matched = col; break

        # ── Pass 3: any synonym keyword present in column header ───────
        if not matched:
            best_kw_score = 0
            for col in remaining_cols:
                cn    = _col_key(col)
                # Count how many keywords appear in this column's normalized name
                score = sum(1 for kw in keywords if kw in cn)
                if score > best_kw_score:
                    best_kw_score = score
                    matched = col
            if best_kw_score == 0:
                matched = None

        # ── Pass 4: data-type inference ────────────────────────────────
        if not matched and len(df) > 3:
            target_type = (
                "amount"      if "Amount" in canonical else
                "date"        if "Date"   in canonical else
                "description" if "Description" in canonical or "Name" in canonical
                              else None
            )
            if target_type == "amount":
                for col in remaining_cols:
                    if _looks_numeric(df[col]):
                        matched = col; break
            elif target_type == "date":
                for col in remaining_cols:
                    if _looks_like_date(df[col]):
                        matched = col; break
            elif target_type == "description":
                for col in remaining_cols:
                    if _looks_like_text(df[col]):
                        matched = col; break

        if matched:
            detected[canonical] = matched
            remaining_cols.remove(matched)
        elif canonical in required:
            missing.append(canonical)

    return detected, missing


# ── All keyword synonyms flattened (used for header-row scoring) ──────────────
_ALL_KEYWORDS = [
    kw
    for keywords in list(BANK_COL_MAP.values()) + list(GL_COL_MAP.values())
    for kw in keywords
] + [_col_key(c) for c in list(BANK_COL_MAP.keys()) + list(GL_COL_MAP.keys())]


def _score_row_as_header(row_values) -> float:
    """
    Score a row as a potential header.
    Returns keyword_matches × (non_empty_ratio)   so sparse metadata rows
    (with only 1-2 filled cells) don't outrank dense real header rows.
    """
    total   = len(row_values)
    non_nan = sum(1 for v in row_values
                  if v is not None and str(v).strip() not in ('', 'nan', 'None'))
    if total == 0 or non_nan == 0:
        return 0.0

    kw_score = 0
    for val in row_values:
        if not isinstance(val, str):
            continue
        key = _col_key(val)
        if not key:
            continue
        # Penalise pure-numeric cells (account numbers, amounts)
        if key.isdigit():
            continue
        if any(kw in key for kw in _ALL_KEYWORDS if len(kw) >= 3):
            kw_score += 1

    density = non_nan / total
    return kw_score * density


def smart_read_file(file_obj, scan_rows: int = 50) -> tuple[pd.DataFrame, int]:
    """
    Read a CSV/Excel file, auto-detecting the true header row by scanning
    the first `scan_rows` rows and scoring each for keyword density.
    Returns (DataFrame with correct header, best_row_index).
    """
    is_csv = hasattr(file_obj, 'name') and file_obj.name.lower().endswith('.csv')

    def _read_raw():
        file_obj.seek(0)
        if is_csv:
            return pd.read_csv(file_obj, header=None, nrows=scan_rows,
                               dtype=str, on_bad_lines='skip')
        else:
            return pd.read_excel(file_obj, header=None, nrows=scan_rows, dtype=str)

    def _read_with_header(hrow: int):
        file_obj.seek(0)
        if is_csv:
            return pd.read_csv(file_obj, header=hrow,
                               dtype=str, on_bad_lines='skip')
        else:
            return pd.read_excel(file_obj, header=hrow, dtype=str)

    try:
        raw = _read_raw()
    except Exception:
        file_obj.seek(0)
        df = pd.read_csv(file_obj, dtype=str) if is_csv \
             else pd.read_excel(file_obj, dtype=str)
        return df, 0

    # Score each row
    scores     = {}
    for i, row in raw.iterrows():
        scores[i] = _score_row_as_header(row.values)

    best_row  = max(scores, key=scores.get)
    best_score = scores[best_row]

    # If nothing scored > 0, fall back to row 0
    if best_score == 0:
        best_row = 0

    try:
        df = _read_with_header(best_row)
    except Exception:
        file_obj.seek(0)
        df = pd.read_csv(file_obj, dtype=str) if is_csv \
             else pd.read_excel(file_obj, dtype=str)
        best_row = 0

    df.dropna(how='all', axis=1, inplace=True)
    df.dropna(how='all', axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, best_row


def fuzzy_contains(vendor: str, bank_desc: str, threshold: int = 70) -> tuple[bool, int]:
    """
    Level-2 'contains' check with fuzzy matching.
    Strategy:
      1. Direct substring check (exact contains) → score 100
      2. Sliding-window partial ratio over bank_desc tokens → rapidfuzz partial_ratio
      3. Token-set ratio for reordered / abbreviated names
    Returns (matched: bool, best_score: int)
    """
    v = normalize(vendor)
    d = normalize(bank_desc)

    if not v or not d:
        return False, 0

    # 1. Exact contains
    if v in d:
        return True, 100

    # 2. Partial ratio (handles truncation well)
    partial = fuzz.partial_ratio(v, d)

    # 3. Token set ratio (handles word reordering / abbreviations)
    token_set = fuzz.token_set_ratio(v, d)

    # 4. WRatio – weighted combination
    wratio = fuzz.WRatio(v, d)

    best = max(partial, token_set, wratio)
    return best >= threshold, best


def reconcile(bank_df: pd.DataFrame,
              gl_df: pd.DataFrame,
              fuzzy_threshold: int,
              tolerance: float) -> pd.DataFrame:
    """
    Simplified 1-to-1 reconciliation:
      L1 – exact amount match (within tolerance)
      L2 – vendor name 'contains' / fuzzy check in bank description to tie-break duplicates
    """
    results = []

    bank_df = bank_df.copy().reset_index(drop=True)
    gl_df   = gl_df.copy().reset_index(drop=True)

    total_rows = len(bank_df)
    progress_bar = st.progress(0)
    status_text = st.empty()
    gl_used = set()   # GL row indices already consumed

    for b_idx, b_row in bank_df.iterrows():
        # Update UI feedback every few rows
        if b_idx % 10 == 0 or b_idx == total_rows - 1:
            progress_bar.progress((b_idx + 1) / total_rows)
            status_text.write(f"⏳ Processing row {b_idx + 1} of {total_rows}...")

        b_amt  = float(b_row["Bank Amount"])
        b_desc = str(b_row.get("Bank Description", ""))

        # ── Level 1: exact amount candidates ──
        target_amt = round(b_amt, 2)
        if tolerance == 0:
            potential_l1 = [i for i in gl_df.index if round(float(gl_df.at[i, "GL Amount"]), 2) == target_amt]
        else:
            potential_l1 = [
                i for i in gl_df.index
                if abs(float(gl_df.at[i, "GL Amount"]) - b_amt) <= tolerance
            ]
        
        l1_hits = [i for i in potential_l1 if i not in gl_used]

        if not l1_hits:
            results.append({
                "Bank Ref":         b_row.get("Bank Ref", b_idx),
                "Bank Date":        b_row.get("Bank Date", ""),
                "Bank Amount":      b_amt,
                "Bank Description": b_desc,
                "GL Vendor":        "-",
                "GL Amount":        "-",
                "Fuzzy Score":      "-",
                "Status":           "Unmatched",
                "GL Row(s)":        "-",
            })
            continue

        # ── Level 2: vendor name fuzzy-contains check to resolve duplicates ──
        best_score   = -1
        best_gl_idx  = None
        desc_matched = False

        for i in l1_hits:
            vendor = str(gl_df.at[i, "GL Vendor Name"])
            matched, score = fuzzy_contains(vendor, b_desc, fuzzy_threshold)
            if matched and score > best_score:
                best_score   = score
                best_gl_idx  = i
                desc_matched = True

        if not desc_matched:
            # Amount matches but description doesn't -> pick the one with highest score to flag
            for i in l1_hits:
                vendor = str(gl_df.at[i, "GL Vendor Name"])
                _, score = fuzzy_contains(vendor, b_desc, fuzzy_threshold)
                if score > best_score:
                    best_score  = score
                    best_gl_idx = i

        # If l1_hits had elements but none scored higher than -1 (edge case)
        if best_gl_idx is None:
            best_gl_idx = l1_hits[0]

        g_row   = gl_df.loc[best_gl_idx]
        gl_used.add(best_gl_idx)
        status  = "Matched" if desc_matched else "Description Mismatch"

        results.append({
            "Bank Ref":         b_row.get("Bank Ref", b_idx),
            "Bank Date":        b_row.get("Bank Date", ""),
            "Bank Amount":      b_amt,
            "Bank Description": b_desc,
            "GL Vendor":        g_row.get("GL Vendor Name", ""),
            "GL Amount":        float(g_row["GL Amount"]),
            "Fuzzy Score":      int(best_score) if best_score >= 0 else "-",
            "Status":           status,
            "GL Row(s)":        str(best_gl_idx),
        })

    progress_bar.empty()
    status_text.empty()
    st.success(f"✅ Reconciliation complete! Processed {total_rows} entries.")
    return pd.DataFrame(results)

def colorize_row(row):
    """Per-row background color for Styler."""
    status = row["Status"]
    if status == "Matched":
        return ["background-color: rgba(63,185,80,0.10)"] * len(row)
    elif status == "Description Mismatch":
        return ["background-color: rgba(248,81,73,0.18)"] * len(row)
    elif "Subset" in status:
        return ["background-color: rgba(227,179,65,0.14)"] * len(row)
    elif status == "Unmatched":
        return ["background-color: rgba(210,168,255,0.12)"] * len(row)
    return [""] * len(row)


def badge(status: str) -> str:
    css = {
        "Matched":              "badge-matched",
        "Description Mismatch": "badge-mismatch",
        "Subset Sum Match":     "badge-subset",
        "Unmatched":            "badge-unmatched",
    }.get(status, "")
    return f'<span class="{css}">{status}</span>'


def load_local_sample_data():
    """Load sample data from the user's specific local path."""
    import os
    base_path = "/Users/rashmirani/Documents/Antigravity Projects/bank_reconciliation_tool/Sample data"
    bank_path = os.path.join(base_path, "Bank st.xls")
    gl_path   = os.path.join(base_path, "Bank GL.XLSX")

    try:
        # Bank loading with smart_read_file logic mock
        with open(bank_path, "rb") as f:
            bank_raw, _ = smart_read_file(f)
        # GL loading
        with open(gl_path, "rb") as f:
            gl_raw, _ = smart_read_file(f)
        return bank_raw, gl_raw
    except Exception as e:
        st.sidebar.warning(f"Note: Local sample files not found or unreadable. Using internal demo data. ({e})")
        return None, None

SAMPLE_BANK_INTERNAL = pd.DataFrame({
    "Bank Ref":         ["TXN001", "TXN002", "TXN003", "TXN004", "TXN005"],
    "Bank Date":        ["2024-03-01", "2024-03-02", "2024-03-02", "2024-03-03", "2024-03-04"],
    "Bank Amount":      [15000.00, 8500.00, 23500.00, 4200.00, 11000.00],
    "Bank Description": [
        "PAYMENT INFOSYS LTD BATCH",
        "TATA CONSULTANCY SVCS INV",
        "WIPRO LIMITED MARCH PYMT",
        "HCL TECHNOLOG 9283",
        "VENDOR UNKNOWN XYZ CORP",
    ],
})

SAMPLE_GL_INTERNAL = pd.DataFrame({
    "GL Vendor Name": ["Infosys Limited", "Tata Consultancy Services",
                       "Wipro Limited", "Wipro Limited",
                       "HCL Technologies", "Reliance Industries"],
    "GL Amount":      [15000.00, 8500.00, 13500.00, 10000.00, 4200.00, 5500.00],
    "GL Date":        ["2024-03-01", "2024-03-02", "2024-03-02",
                       "2024-03-02", "2024-03-03", "2024-03-04"],
    "GL Reference":   ["GL-001", "GL-002", "GL-003", "GL-004", "GL-005", "GL-006"],
})


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px'>
      <div style='font-size:2.5rem'>🏦</div>
      <div style='font-size:1.3rem; font-weight:700; color:#58a6ff'>BankRecon Pro</div>
      <div style='font-size:0.75rem; color:#8b949e; margin-top:4px'>Two-Level Reconciliation Engine</div>
    </div>
    <hr style='border-color:#30363d; margin:12px 0'>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Parameters")

    fuzzy_threshold = st.slider(
        "Fuzzy Match Threshold (%)",
        min_value=50, max_value=100, value=70, step=5,
        help="Minimum similarity score for vendor name → bank description match. "
             "Lower = more lenient, higher = stricter."
    )

    tolerance = st.number_input(
        "Amount Tolerance (₹ / $)",
        min_value=0.0, max_value=100.0, value=0.01, step=0.01,
        format="%.2f",
        help="Acceptable difference between Bank Amount and GL Amount for Level-1 match."
    )

    st.markdown("---")
    st.markdown("### 📖 Status Legend")
    st.markdown("""
    <div style='font-size:0.82rem; line-height:2'>
      <span class='badge-matched'>Matched</span> — L1 + L2 pass<br>
      <span class='badge-mismatch'>Description Mismatch</span> — L1 pass, L2 fail → flag for audit<br>
      <span class='badge-subset'>Subset Sum Match</span> — One bank entry = sum of multiple GL lines<br>
      <span class='badge-unmatched'>Unmatched</span> — No GL counterpart found
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 How It Works")
    st.markdown("""
    <div style='font-size:0.8rem; color:#8b949e; line-height:1.7'>
      <b>Level 1</b> — Exact Amount isolation<br>
      <b>Level 2</b> — Fuzzy vendor-name 'contains' check using:<br>
      &nbsp;&nbsp;• <code>partial_ratio</code> (truncation)<br>
      &nbsp;&nbsp;• <code>token_set_ratio</code> (word reorder)<br>
      &nbsp;&nbsp;• <code>WRatio</code> (weighted blend)<br>
      <b>Subset Sum</b> — Combinatorial search up to 4 GL lines
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Main area
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Bank Reconciliation Engine</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e; margin-top:-6px'>Two-level verification · Fuzzy matching · Subset-sum detection</p>", unsafe_allow_html=True)

tabs = st.tabs(["📥 Upload & Reconcile", "📊 Results", "📋 Audit Report", "ℹ️ Logic Guide"])


# ── TAB 1: UPLOAD ──────────────────────────────────────────────────────────────
with tabs[0]:
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🏦 Bank Statement")
        bank_file = st.file_uploader(
            "Upload Bank CSV / Excel",
            type=["csv", "xlsx", "xls"],
            key="bank_file",
            help="Required columns: Bank Ref, Bank Date, Bank Amount, Bank Description"
        )
        with st.expander("📌 Required columns & format"):
            st.markdown("""
            | Column | Type | Example |
            |--------|------|---------|
            | `Bank Ref` | str | TXN001 |
            | `Bank Date` | date | 2024-03-01 |
            | `Bank Amount` | float | 15000.00 |
            | `Bank Description` | str | PAYMENT INFOSYS LTD |
            """)

    with col2:
        st.markdown("#### 📒 General Ledger")
        gl_file = st.file_uploader(
            "Upload GL CSV / Excel",
            type=["csv", "xlsx", "xls"],
            key="gl_file",
            help="Required columns: GL Vendor Name, GL Amount, GL Date, GL Reference"
        )
        with st.expander("📌 Required columns & format"):
            st.markdown("""
            | Column | Type | Example |
            |--------|------|---------|
            | `GL Vendor Name` | str | Infosys Limited |
            | `GL Amount` | float | 15000.00 |
            | `GL Date` | date | 2024-03-01 |
            | `GL Reference` | str | GL-001 |
            """)

    st.markdown("---")

    # Load data
    use_sample = False
    bank_df = None
    gl_df   = None

    BANK_REQUIRED = ["Bank Amount", "Bank Description"]
    GL_REQUIRED   = ["GL Vendor Name", "GL Amount"]

    # Initialize session state for data persistence
    if "bank_df" not in st.session_state: st.session_state["bank_df"] = None
    if "gl_df" not in st.session_state:   st.session_state["gl_df"] = None
    if "bank_detected" not in st.session_state: st.session_state["bank_detected"] = {}
    if "gl_detected" not in st.session_state:   st.session_state["gl_detected"] = {}

    def _load_with_picker(file_obj, col_map, required, file_label, key_prefix):
        """
        Load a file with smart_read_file, then show an always-visible
        raw preview + header-row picker so the user can correct detection.
        Returns (df, detected, missing).
        """
        raw, hdr_row = smart_read_file(file_obj)
        detected, missing = auto_map_columns(raw, col_map, required)

        # ── Always show raw preview + row picker ──────────────────────
        file_obj.seek(0)
        try:
            preview = (pd.read_csv(file_obj, header=None, nrows=50, dtype=str,
                                   on_bad_lines='skip')
                       if file_obj.name.lower().endswith('.csv')
                       else pd.read_excel(file_obj, header=None, nrows=50, dtype=str))
        except Exception:
            preview = pd.DataFrame()

        auto_ok = not missing
        label   = (f"✅ {file_label} — header auto-detected at row {hdr_row}"
                   if auto_ok
                   else f"⚠️ {file_label} — header detected at row {hdr_row} "
                        f"(missing: `{'`, `'.join(missing)}`)")

        with st.expander(label, expanded=not auto_ok):
            if not preview.empty:
                st.markdown("**Raw file preview (row in blue is the suspected header):**")
                def _highlight_hdr(s):
                    return ["background-color: rgba(31,111,235,0.2)" if s.name == hdr_row else "" for _ in s]
                st.dataframe(preview.style.apply(_highlight_hdr, axis=1), use_container_width=True, height=200)

            chosen = st.number_input(
                f"Select header row for {file_label}:",
                min_value=0, max_value=max(len(preview)-1, 0), value=int(hdr_row), step=1,
                key=f"{key_prefix}_hdr_picker"
            )

            if chosen != hdr_row or missing:
                file_obj.seek(0)
                try:
                    raw = (pd.read_csv(file_obj, header=int(chosen), dtype=str, on_bad_lines='skip')
                           if file_obj.name.lower().endswith('.csv')
                           else pd.read_excel(file_obj, header=int(chosen), dtype=str))
                    raw.dropna(how='all', axis=1, inplace=True)
                    raw.dropna(how='all', axis=0, inplace=True)
                    raw.reset_index(drop=True, inplace=True)
                    detected, missing = auto_map_columns(raw, col_map, required)
                except Exception as e:
                    st.error(f"Error reading row {chosen}: {e}")

        return raw, detected, missing

    if bank_file:
        raw, det, miss = _load_with_picker(bank_file, BANK_COL_MAP, BANK_REQUIRED, "Bank Statement", "bank")
        st.session_state["bank_df"] = raw
        st.session_state["bank_detected"] = det

    if gl_file:
        raw, det, miss = _load_with_picker(gl_file, GL_COL_MAP, GL_REQUIRED, "General Ledger", "gl")
        st.session_state["gl_df"] = raw
        st.session_state["gl_detected"] = det

    bank_df = st.session_state["bank_df"]
    gl_df   = st.session_state["gl_df"]
    bank_detected = st.session_state["bank_detected"]
    gl_detected   = st.session_state["gl_detected"]

    if bank_df is None or gl_df is None:
        st.info("💡 No files uploaded yet — click below to run with **local sample data**.")
        if st.button("🚀 Run with Sample Data", use_container_width=True, type="primary"):
            lb, lg = load_local_sample_data()
            if lb is not None:
                st.session_state["bank_df"] = lb
                st.session_state["gl_df"]   = lg
                det_b, _ = auto_map_columns(lb, BANK_COL_MAP, BANK_REQUIRED)
                det_g, _ = auto_map_columns(lg, GL_COL_MAP, GL_REQUIRED)
                st.session_state["bank_detected"] = det_b
                st.session_state["gl_detected"]   = det_g
                st.rerun()
            else:
                st.session_state["bank_df"] = SAMPLE_BANK_INTERNAL.copy()
                st.session_state["gl_df"]   = SAMPLE_GL_INTERNAL.copy()
                st.session_state["bank_detected"] = {c: c for c in BANK_COL_MAP}
                st.session_state["gl_detected"]   = {c: c for c in GL_COL_MAP}
                st.rerun()
    else:
        # ─────────────────────────────────────────────────────────────
        #  Interactive column mapper — always shown for uploaded files
        # ─────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🗺️ Column Mapping")
        st.markdown(
            "<p style='color:#8b949e; font-size:0.87rem'>Auto-detection has pre-filled the dropdowns below. "
            "If any mapping looks wrong, change it using the dropdown — then click <b>Apply & Reconcile</b>.</p>",
            unsafe_allow_html=True,
        )

        # Helper: return the best pre-selected value for a selectbox
        def _best_option(col_options, detected_map, canon):
            auto = detected_map.get(canon)
            # Use auto-detected column if it's still in the options list
            if auto and auto in col_options:
                return col_options.index(auto)
            return 0   # fall back to first option ("-- not mapped --")

        NONE_OPT = "-- not mapped --"

        bank_cols_raw = bank_df.columns.tolist() if bank_df is not None else []
        gl_cols_raw   = gl_df.columns.tolist()   if gl_df   is not None else []

        bank_opts = [NONE_OPT] + bank_cols_raw
        gl_opts   = [NONE_OPT] + gl_cols_raw

        mc1, mc2 = st.columns(2, gap="large")

        # ── Bank Statement mapping ──────────────────────────────────
        with mc1:
            st.markdown("#### 🏦 Bank Statement")
            bank_map_ui = {}
            bank_required_fields = {
                "Bank Ref":         ("optional", "🔵"),
                "Bank Date":        ("optional", "🔵"),
                "Bank Amount":      ("required", "🔴"),
                "Bank Description": ("required", "🔴"),
            }
            for canon, (req, icon) in bank_required_fields.items():
                label = f"{icon} {canon} {'*(required)*' if req == 'required' else '*(optional)*'}"
                default_idx = _best_option(bank_opts, bank_detected, canon)
                chosen = st.selectbox(
                    label,
                    options=bank_opts,
                    index=default_idx,
                    key=f"bank_map_{canon}",
                )
                bank_map_ui[canon] = chosen if chosen != NONE_OPT else None

        # ── GL mapping ─────────────────────────────────────────────
        with mc2:
            st.markdown("#### 📒 General Ledger")
            gl_map_ui = {}
            gl_required_fields = {
                "GL Vendor Name": ("required", "🔴"),
                "GL Amount":      ("required", "🔴"),
                "GL Date":        ("optional", "🔵"),
                "GL Reference":   ("optional", "🔵"),
            }
            for canon, (req, icon) in gl_required_fields.items():
                label = f"{icon} {canon} {'*(required)*' if req == 'required' else '*(optional)*'}"
                default_idx = _best_option(gl_opts, gl_detected, canon)
                chosen = st.selectbox(
                    label,
                    options=gl_opts,
                    index=default_idx,
                    key=f"gl_map_{canon}",
                )
                gl_map_ui[canon] = chosen if chosen != NONE_OPT else None

        # ── Apply user mapping overrides ─────────────────────────────
        # Check for required fields still unmapped
        ui_bank_missing = [c for c, v in bank_map_ui.items()
                           if v is None and c in BANK_REQUIRED]
        ui_gl_missing   = [c for c, v in gl_map_ui.items()
                           if v is None and c in GL_REQUIRED]

        if ui_bank_missing or ui_gl_missing:
            if ui_bank_missing:
                st.error(
                    f"❌ Please map the required Bank column(s): "
                    f"`{'`, `'.join(ui_bank_missing)}`"
                )
            if ui_gl_missing:
                st.error(
                    f"❌ Please map the required GL column(s): "
                    f"`{'`, `'.join(ui_gl_missing)}`"
                )
        else:
            # Build renamed DataFrames from user selections
            bank_rename = {v: k for k, v in bank_map_ui.items() if v}
            gl_rename   = {v: k for k, v in gl_map_ui.items()   if v}
            bank_df = bank_df.rename(columns=bank_rename)
            gl_df   = gl_df.rename(columns=gl_rename)

            st.success("✅ Column mapping looks good. Ready to reconcile.")
            if st.button("▶️ Apply & Reconcile", use_container_width=True,
                         type="primary", key="run_btn_manual"):
                pass   # falls through to reconcile block below

    # Preview (always show after files loaded)
    if bank_df is not None and gl_df is not None:
        prev1, prev2 = st.columns(2)
        with prev1:
            st.markdown(f"**Bank Statement** — {len(bank_df)} rows")
            st.dataframe(bank_df, use_container_width=True, height=220)
        with prev2:
            st.markdown(f"**General Ledger** — {len(gl_df)} rows")
            st.dataframe(gl_df, use_container_width=True, height=220)

    # Run reconciliation — only if no missing required columns
    _ui_bank_missing = ui_bank_missing if 'ui_bank_missing' in dir() else []
    _ui_gl_missing   = ui_gl_missing   if 'ui_gl_missing'   in dir() else []

    if (bank_df is not None and gl_df is not None
            and not _ui_bank_missing and not _ui_gl_missing
            and "Bank Amount" in bank_df.columns
            and "GL Amount" in gl_df.columns
            and "GL Vendor Name" in gl_df.columns
            and "Bank Description" in bank_df.columns):
        with st.spinner("Running reconciliation engine…"):
            result_df = reconcile(bank_df, gl_df, fuzzy_threshold, tolerance)
        st.session_state["result_df"] = result_df
        st.session_state["bank_df"]   = bank_df
        st.session_state["gl_df"]     = gl_df

        if use_sample or bank_file or gl_file:
            st.success("✅ Reconciliation complete! Switch to the **Results** or **Audit Report** tab.")


# ── TAB 2: RESULTS ─────────────────────────────────────────────────────────────
with tabs[1]:
    if "result_df" not in st.session_state:
        st.info("Run reconciliation in the Upload tab first.")
    else:
        df = st.session_state["result_df"]

        # KPI metrics
        total      = len(df)
        matched    = (df["Status"] == "Matched").sum()
        mismatch   = (df["Status"] == "Description Mismatch").sum()
        subset     = (df["Status"] == "Subset Sum Match").sum()
        unmatched  = (df["Status"] == "Unmatched").sum()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Bank Entries", total)
        c2.metric("✅ Matched",          matched,   delta=f"{matched/total*100:.0f}%")
        c3.metric("🔴 Desc Mismatch",    mismatch,  delta=f"-{mismatch/total*100:.0f}%" if mismatch else "0%")
        c4.metric("🟡 Subset Sum",       subset)
        c5.metric("🟣 Unmatched",        unmatched)

        st.markdown("---")

        # Filters
        fcol1, fcol2 = st.columns([2, 1])
        with fcol1:
            filter_status = st.multiselect(
                "Filter by Status",
                options=df["Status"].unique().tolist(),
                default=df["Status"].unique().tolist(),
                key="filter_status",
            )
        with fcol2:
            search_term = st.text_input("🔍 Search vendor / description", key="search_term")

        filtered = df[df["Status"].isin(filter_status)]
        if search_term:
            mask = (
                filtered["GL Vendor"].str.contains(search_term, case=False, na=False) |
                filtered["Bank Description"].str.contains(search_term, case=False, na=False)
            )
            filtered = filtered[mask]

        st.markdown(f"**Showing {len(filtered)} of {len(df)} entries**")

        # Styled results table
        styled = filtered.style.apply(colorize_row, axis=1)
        st.dataframe(styled, use_container_width=True, height=400)

        # Download
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        st.download_button(
            "⬇️ Download Full Results (Excel)",
            data=buf,
            file_name="reconciliation_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


# ── TAB 3: AUDIT REPORT ────────────────────────────────────────────────────────
with tabs[2]:
    if "result_df" not in st.session_state:
        st.info("Run reconciliation in the Upload tab first.")
    else:
        df = st.session_state["result_df"]
        mismatches = df[df["Status"] == "Description Mismatch"].copy()
        subsets    = df[df["Status"] == "Subset Sum Match"].copy()
        unmatched  = df[df["Status"] == "Unmatched"].copy()

        # ── Description Mismatches ──
        st.markdown("### 🔴 Description Mismatches — Manual Audit Required")
        st.markdown(
            '<p style="color:#8b949e; font-size:0.87rem">Amount matched (Level 1) but vendor name '
            'could NOT be found in bank description (Level 2). These require manual review.</p>',
            unsafe_allow_html=True
        )
        if mismatches.empty:
            st.success("🎉 No description mismatches found!")
        else:
            # Build HTML table with red highlights
            html_rows = ""
            for _, r in mismatches.iterrows():
                html_rows += f"""
                <tr style='background:rgba(248,81,73,0.12)'>
                  <td style='padding:8px 12px; border-bottom:1px solid #30363d'>{r['Bank Ref']}</td>
                  <td style='padding:8px 12px; border-bottom:1px solid #30363d'>{r['Bank Date']}</td>
                  <td style='padding:8px 12px; border-bottom:1px solid #30363d; font-weight:600'>
                      ₹ {float(r['Bank Amount']):,.2f}</td>
                  <td style='padding:8px 12px; border-bottom:1px solid #30363d; color:#f85149'>
                      {r['Bank Description']}</td>
                  <td style='padding:8px 12px; border-bottom:1px solid #30363d'>{r['GL Vendor']}</td>
                  <td style='padding:8px 12px; border-bottom:1px solid #30363d'>{r['Fuzzy Score']}</td>
                  <td style='padding:8px 12px; border-bottom:1px solid #30363d'>
                      <span class='badge-mismatch'>Description Mismatch</span></td>
                </tr>"""

            st.markdown(f"""
            <div style='border:1px solid #f85149; border-radius:10px; overflow:hidden; margin-bottom:20px'>
              <table style='width:100%; border-collapse:collapse; font-size:0.85rem'>
                <thead>
                  <tr style='background:#21262d; color:#58a6ff'>
                    <th style='padding:10px 12px; text-align:left'>Bank Ref</th>
                    <th style='padding:10px 12px; text-align:left'>Date</th>
                    <th style='padding:10px 12px; text-align:left'>Amount</th>
                    <th style='padding:10px 12px; text-align:left'>Bank Description ⚠️</th>
                    <th style='padding:10px 12px; text-align:left'>GL Vendor</th>
                    <th style='padding:10px 12px; text-align:left'>Score</th>
                    <th style='padding:10px 12px; text-align:left'>Status</th>
                  </tr>
                </thead>
                <tbody>{html_rows}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

        # ── Subset Sum Matches ──
        st.markdown("### 🟡 Subset Sum Matches")
        st.markdown(
            '<p style="color:#8b949e; font-size:0.87rem">One bank entry covers multiple GL lines. '
            'Verify the GL entry breakdown below.</p>',
            unsafe_allow_html=True
        )
        if subsets.empty:
            st.success("No subset-sum entries found.")
        else:
            st.dataframe(subsets.style.apply(colorize_row, axis=1), use_container_width=True)

        # ── Unmatched ──
        st.markdown("### 🟣 Unmatched Bank Entries")
        if unmatched.empty:
            st.success("All bank entries have a GL counterpart!")
        else:
            st.dataframe(unmatched.style.apply(colorize_row, axis=1), use_container_width=True)

        # Audit download
        buf2 = io.BytesIO()
        with pd.ExcelWriter(buf2, engine="openpyxl") as writer:
            mismatches.to_excel(writer, sheet_name="Desc_Mismatches", index=False)
            subsets.to_excel(writer,    sheet_name="Subset_Sum",       index=False)
            unmatched.to_excel(writer,  sheet_name="Unmatched",        index=False)
        buf2.seek(0)
        st.download_button(
            "⬇️ Download Audit Report (Excel)",
            data=buf2,
            file_name="audit_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


# ── TAB 4: LOGIC GUIDE ──────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("## 🧠 Reconciliation Logic Guide")

    st.markdown("""
    <div class='info-box'>
    <h4 style='color:#58a6ff; margin-top:0'>🔵 Level 1 — Amount Isolation</h4>
    <p style='color:#c9d1d9'>Every bank transaction is first matched against GL entries by exact amount
    (within your configured tolerance). Only GL rows within that tolerance band proceed to Level 2.</p>
    <code style='background:#0d1117; padding:4px 8px; border-radius:4px; font-size:0.85rem'>
      |Bank Amount − GL Amount| ≤ tolerance
    </code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    <h4 style='color:#58a6ff; margin-top:0'>🟢 Level 2 — Vendor Name Fuzzy Contains</h4>
    <p style='color:#c9d1d9'>Three complementary fuzzy algorithms are applied in parallel; the highest score wins:</p>
    <ul style='color:#c9d1d9'>
      <li><b>partial_ratio</b> — Sliding window; great for <em>truncated</em> names
          (e.g. <code>INFOSYS LI</code> → <code>Infosys Limited</code>)</li>
      <li><b>token_set_ratio</b> — Word-level set comparison; handles <em>reordered</em> words
          (e.g. <code>CONSULTANCY TATA</code> → <code>Tata Consultancy Services</code>)</li>
      <li><b>WRatio</b> — Weighted blend of multiple metrics; robust fallback</li>
    </ul>
    <p style='color:#c9d1d9'>If best score ≥ threshold → <b>Matched</b>; else → <b>Description Mismatch</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    <h4 style='color:#58a6ff; margin-top:0'>🟡 Subset-Sum Detection</h4>
    <p style='color:#c9d1d9'>When no single GL line matches the bank amount, a combinatorial search
    checks whether a <em>subset</em> of remaining GL entries sums to the bank amount.
    Searches subsets of size 2–4 to stay efficient.</p>
    <pre style='background:#0d1117; padding:10px; border-radius:6px; font-size:0.82rem; color:#e6edf3'>
for size in [2, 3, 4]:
    for combo in combinations(gl_entries, size):
        if |sum(combo) - bank_amount| ≤ tolerance:
            → flag as Subset Sum Match</pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    <h4 style='color:#58a6ff; margin-top:0'>📐 Tuning Your Threshold</h4>

    | Threshold | Behaviour | Best for |
    |-----------|-----------|----------|
    | 50–60 % | Very lenient — catches highly truncated / abbreviated names | Short bank descriptions |
    | 70 % *(default)* | Balanced — good for most bank formats | General use |
    | 85–95 % | Strict — only near-exact matches | Clean, unabbreviated descriptions |
    | 100 % | Exact substring only | Perfect data |

    </div>
    """, unsafe_allow_html=True)
