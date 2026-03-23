"""
build_sector_map.py — Build Russell 1000 sector map and write pattern_engine/sector.py

Steps:
  1. Fetch S&P 500 from Wikipedia (503 stocks with GICS sectors)
  2. Curated list of ~500 known Russell 1000 non-S&P500 mid-caps
  3. Bulk-validate mid-caps via yfinance (fast group download)
  4. Write pattern_engine/sector.py

Run once, commit the output.
"""
from __future__ import annotations
import io
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "pattern_engine" / "sector.py"

# ── GICS → internal sector ────────────────────────────────────────────────────
GICS_TO_INTERNAL: dict[str, str] = {
    "Information Technology":  "Tech",
    "Communication Services":  "Tech",
    "Financials":              "Finance",
    "Real Estate":             "Finance",
    "Health Care":             "Health",
    "Consumer Discretionary":  "Consumer",
    "Consumer Staples":        "Consumer",
    "Industrials":             "Industrial",
    "Materials":               "Industrial",
    "Utilities":               "Industrial",
    "Energy":                  "Energy",
}

# ── Curated Russell 1000 mid-cap additions (non-S&P500) ──────────────────────
# Organized by our internal sector; first-assignment wins if a ticker appears twice.
MIDCAP_CANDIDATES: dict[str, list[str]] = {
    "Tech": [
        # Software / Cloud
        "DDOG", "SNOW", "NET", "CRWD", "ZS", "OKTA", "FTNT", "HUBS", "PAYC", "PCTY",
        "VEEV", "EPAM", "NTNX", "PSTG", "GWRE", "TYL", "BL", "MANH", "WEX", "IT",
        "EXLS", "DXC", "CTSH", "PRFT", "CDAY", "SPSC", "NSIT", "PLUS", "TASK",
        "APPN", "JAMF", "ZETA", "KVYO", "BRZE", "MSTR", "NCNO", "SYM", "TTEK",
        # Hardware / Semiconductors
        "HPE", "NTAP", "STX", "WDC", "JNPR", "SWKS", "MPWR", "ON", "PTC", "ANSS",
        "CDW", "SMCI", "LOGI", "ZBRA", "CGNX", "TRMB", "AKAM", "FFIV", "KEYS",
        # Defense IT
        "LDOS", "SAIC", "CACI", "BAH",
        # Communication Services (media/adtech → Tech bucket)
        "IAC", "BMBL", "RBLX", "WMG", "LYV", "NWSA", "NWS", "OMC", "IPG",
        "TTWO", "EA", "MTCH", "ZD", "FWONK", "PARAA", "AMCX", "IHRT",
    ],
    "Finance": [
        # Insurance
        "HIG", "LNC", "UNM", "AFG", "RNR", "FAF", "MKL", "RLI", "KNSL", "RYAN",
        "BRP", "ERIE", "VOYA", "JXN", "FG", "GL", "RGA",
        # Banks / Regional
        "ALLY", "WAL", "COLB", "PNFP", "CVBF", "BANR", "UMPQ", "IBOC", "SBCF",
        "FHN", "KEY", "ZION", "CMA", "HBAN", "RF",
        # Fintech / Consumer Finance
        "HOOD", "SOFI", "LC", "UPST", "AFRM", "SLM", "ESNT", "NMI",
        "COOP", "RKT", "PFSI", "UWMC", "TREE", "ENVA", "WRLD",
        # Asset Management / Exchanges
        "IVZ", "AMG", "APAM", "VCTR", "MORN", "SEIC", "NDAQ", "MKTX", "CBOE",
        # Auto Dealers
        "LAD", "AN", "PAG", "ABG", "GPI",
        # REITs
        "CUBE", "NSA", "REXR", "COLD", "TRNO", "LXP", "IIPR", "JBGS", "EQC",
        "OHI", "SBRA", "NHI", "LTC", "BRT", "SKT", "DEI", "PDM", "CLI",
        "AIRC", "NXRT", "IRT", "UDR", "ELS", "SUI", "OUT", "LAMR", "CCO",
        "INN", "APLE", "CLDT", "PK", "RHP", "AMH",
    ],
    "Health": [
        # Biotech
        "ALNY", "BMRN", "EXAS", "IONS", "JAZZ", "NVCR", "ACAD", "INCY",
        "ARWR", "BEAM", "RARE", "SRPT", "FOLD", "RPRX", "ITCI", "SAGE",
        "APLS", "KRYS", "RCUS", "IMVT", "ROIV", "TARS", "PTCT", "NKTR",
        "KYMR", "ZYME", "HRMY", "AGIO", "ALLO", "BCAB", "NVAX",
        # Medical Devices
        "IRTC", "NTRA", "MMSI", "IART", "LMAT", "LIVN", "NXGN", "RGEN",
        "PODD", "XRAY", "ALGN", "KIDS", "QDEL", "FLGT", "CDNA",
        # Diagnostics / Distribution
        "HALO", "PRGO", "PCRX", "NEOG", "GH", "PDCO", "HSIC",
        # Healthcare Services
        "AMED", "HQY", "CHE", "ENSG", "SGRY", "ADUS", "OPCH", "MMS",
        "ACCD", "OSCR", "NTRA", "CERT",
    ],
    "Consumer": [
        # Apparel / Footwear
        "ONON", "SKX", "CROX", "COLM", "HBI", "PVH", "CPRI", "SIG", "TPR",
        "VFC", "WWW", "GOOS", "RVLV", "BOOT", "HELE",
        # Home / E-commerce
        "W", "RH", "ARHS",
        # Gaming / Entertainment
        "DKNG", "BYD", "TTWO", "EA",
        # Restaurants
        "TXRH", "SHAK", "WING", "JACK", "DENN", "BROS", "DNUT",
        # Specialty Retail
        "LKQ", "BURL", "FIVE", "OLLI", "BIG", "TLYS",
        # Leisure / Recreation
        "PII", "THO", "WGO", "MODG", "GOLF", "ACUSHNET", "PTON", "PLNT",
        # Food Distribution
        "USFD", "PFGC", "CHEF",
        # Beverages / Specialty
        "FIZZ", "CELH", "COKE", "DNUT",
        # Staples - Food
        "SJM", "HRL", "CAG", "CPB", "INGR", "LANC", "FRPT", "BRBR", "SMPL",
        "NOMD", "VITL",
        # Staples - Personal Care
        "COTY", "ELF", "USNA", "HIMS", "SFM", "GO", "WEIS",
    ],
    "Industrial": [
        # Aerospace / Defense
        "HEI", "DRS", "TDG", "HWM", "TXT", "AXON", "MOOG",
        # Construction / Engineering
        "PWR", "MTZ", "ROAD", "MYR", "STRL", "MYRG", "GLDD", "ABM", "VSE",
        "GVA", "PRIM", "IESC",
        # Transport / Logistics
        "SAIA", "RXO", "XPO", "CVLG", "HTLD", "MARTEN", "ARCB",
        "GATX", "AER", "AIR", "WLFC",
        # Industrial Machinery / Equipment
        "GNRC", "CSL", "ESAB", "FLOW", "ENPRO", "CSWI", "ATKR", "IIIN",
        # Metals / Mining / Building Materials
        "WOR", "HAYN", "KALU", "CENX", "AMR", "ARCH", "CEIX",
        "EXP", "SUM", "USCR", "WMS", "ATI",
        # Staffing / Business Services
        "RHI", "ASGN", "TNET", "NSP", "BFAM", "KFRC", "KELYA",
        # Environmental / Waste
        "CWST", "ASTE", "NVRI",
        # Utilities
        "PNW", "OTTR", "NWE", "AVA", "IDACORP", "SWX", "SJW", "AWR", "MSEX",
        "AES", "NRG", "VST", "BKH",
    ],
    "Energy": [
        # E&P
        "AR", "RRC", "SM", "CNX", "CIVI", "OVV", "CHRD", "NOG", "VTLE",
        "MUR", "CRC", "TALO",
        # Oilfield Services
        "FTI", "HP", "PTEN", "LBRT", "PUMP", "NEX", "TDW", "OIS", "DNOW",
        # Midstream / MLPs
        "AM", "CPLP", "USAC", "WES", "DKL", "ENLC", "CEQP",
    ],
}

# ── Sector proxies (one ETF/stock per sector for regime baseline) ─────────────
SECTOR_PROXIES: dict[str, str] = {
    "Index":      "SPY",
    "Tech":       "QQQ",
    "Finance":    "JPM",
    "Health":     "UNH",
    "Consumer":   "WMT",
    "Industrial": "CAT",
    "Energy":     "XOM",
}


def fetch_sp500() -> dict[str, str]:
    """Fetch S&P 500 tickers + GICS sectors from Wikipedia."""
    print("Fetching S&P 500 from Wikipedia...")
    req = urllib.request.Request(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8")
    df = pd.read_html(io.StringIO(html))[0][["Symbol", "GICS Sector"]]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    result = {
        row.Symbol: GICS_TO_INTERNAL.get(row["GICS Sector"], "Industrial")
        for _, row in df.iterrows()
    }
    print(f"  S&P 500: {len(result)} tickers")
    return result


def validate_midcaps(candidates: list[str], batch_size: int = 100) -> list[str]:
    """Bulk-validate tickers via yfinance 5-day download. Fast: 1 call per batch."""
    print(f"\nValidating {len(candidates)} mid-cap candidates via yfinance...")
    valid: list[str] = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        try:
            data = yf.download(
                batch, period="5d", auto_adjust=True,
                progress=False, threads=True,
            )
            if len(batch) == 1:
                cols = batch if not data.empty else []
            elif "Close" in data.columns:
                cols = data["Close"].dropna(how="all", axis=1).columns.tolist()
            else:
                cols = []
            valid.extend(cols)
            print(f"  Batch {i // batch_size + 1}: {len(cols)}/{len(batch)} valid")
        except Exception as exc:
            print(f"  Batch {i // batch_size + 1} error: {exc}")
    print(f"Valid mid-cap tickers: {len(valid)}")
    return valid


def build_sector_map() -> dict[str, str]:
    sp500 = fetch_sp500()

    # Flatten mid-cap candidates, skip any already in S&P 500
    candidate_map: dict[str, str] = {}
    for sector, tickers in MIDCAP_CANDIDATES.items():
        for t in tickers:
            t = t.upper().strip()
            if t not in sp500 and t not in candidate_map:
                candidate_map[t] = sector

    print(f"  Mid-cap new candidates: {len(candidate_map)}")
    valid = validate_midcaps(list(candidate_map.keys()))

    sector_map = dict(sp500)  # S&P 500 first (authoritative sectors)
    for t in valid:
        if t not in sector_map:
            sector_map[t] = candidate_map[t]

    # Always include sector proxy ETFs
    sector_map["SPY"] = "Index"
    sector_map["QQQ"] = "Index"

    return sector_map


def write_sector_py(sector_map: dict[str, str]) -> None:
    tickers = sorted(sector_map.keys())
    dist = Counter(sector_map.values())
    lines = [
        '"""',
        'sector.py — FPPE ticker universe and sector mapping.',
        '',
        f'Universe: {len(tickers)} tickers (Russell 1000 scope)',
        'Sectors: Index, Tech, Finance, Health, Consumer, Industrial, Energy',
        '',
        'Regenerate: python scripts/build_sector_map.py',
        '"""',
        'from __future__ import annotations',
        '',
        'SECTOR_MAP: dict[str, str] = {',
    ]
    # Group by sector for readability
    by_sector: dict[str, list[str]] = {}
    for t in tickers:
        s = sector_map[t]
        by_sector.setdefault(s, []).append(t)
    for sector in ["Index", "Tech", "Finance", "Health", "Consumer", "Industrial", "Energy"]:
        grp = by_sector.get(sector, [])
        if not grp:
            continue
        lines.append(f"    # {sector} ({len(grp)})")
        for t in grp:
            lines.append(f'    "{t}": "{sector}",')
    lines += [
        '}',
        '',
        'TICKERS: list[str] = list(SECTOR_MAP.keys())',
        '',
        'SECTOR_PROXIES: dict[str, str] = {',
    ]
    for s, proxy in SECTOR_PROXIES.items():
        lines.append(f'    "{s}": "{proxy}",')
    lines += [
        '}',
        '',
        '',
        'def compute_sector_features(db):  # type: ignore[override]',
        '    """Placeholder -- sector relative features deferred to M9 data pipeline.',
        '    Returns db unchanged until sector feature engineering is implemented.',
        '    """',
        '    return db',
    ]
    content = "\n".join(lines) + "\n"
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}")
    print(f"Total tickers: {len(tickers)}")
    for s in ["Index", "Tech", "Finance", "Health", "Consumer", "Industrial", "Energy"]:
        print(f"  {s}: {dist.get(s, 0)}")


if __name__ == "__main__":
    sector_map = build_sector_map()
    write_sector_py(sector_map)
    print("\nDone. Run: python -m pytest tests/ -q -m 'not slow'")
