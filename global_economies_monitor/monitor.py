#!/usr/bin/env python3
"""
Global Economies Monitor
Tracks GDP growth, inflation, unemployment, stock market indices, and forex rates.

Data sources (when --live flag is used):
  - World Bank API  (economic indicators)
  - yfinance        (stock market indices)
  - open.er-api.com (forex rates)

Default mode ships with curated 2024-2025 data so the dashboard works offline.
"""

import sys
import time
import argparse
import threading
import requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box
from rich.rule import Rule

console = Console()

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

COUNTRIES: dict[str, str] = {
    "US": "United States",
    "CN": "China",
    "JP": "Japan",
    "DE": "Germany",
    "GB": "United Kingdom",
    "IN": "India",
    "FR": "France",
    "BR": "Brazil",
    "CA": "Canada",
    "AU": "Australia",
    "KR": "South Korea",
    "MX": "Mexico",
    "ID": "Indonesia",
    "SA": "Saudi Arabia",
    "ZA": "South Africa",
    "RU": "Russia",
    "IT": "Italy",
    "ES": "Spain",
    "AR": "Argentina",
    "TR": "Turkey",
}

STOCK_INDICES: dict[str, tuple[str, str]] = {
    "^GSPC":     ("S&P 500",        "US"),
    "^DJI":      ("Dow Jones",      "US"),
    "^IXIC":     ("NASDAQ",         "US"),
    "^FTSE":     ("FTSE 100",       "GB"),
    "^N225":     ("Nikkei 225",     "JP"),
    "^HSI":      ("Hang Seng",      "HK"),
    "^DAX":      ("DAX",            "DE"),
    "^CAC40":    ("CAC 40",         "FR"),
    "^AXJO":     ("ASX 200",        "AU"),
    "^BSESN":    ("BSE Sensex",     "IN"),
    "^KS11":     ("KOSPI",          "KR"),
    "^BVSP":     ("Bovespa",        "BR"),
    "^STOXX50E": ("Euro Stoxx 50",  "EU"),
}

BASE_FOREX_CURRENCY = "USD"
FOREX_TARGETS = [
    "EUR", "GBP", "JPY", "CNY", "INR", "BRL",
    "CAD", "AUD", "KRW", "MXN", "CHF", "SGD", "TRY", "ZAR",
]

WORLD_BANK_INDICATORS: dict[str, str] = {
    "gdp_growth":    "NY.GDP.MKTP.KD.ZG",
    "inflation":     "FP.CPI.TOTL.ZG",
    "unemployment":  "SL.UEM.TOTL.ZS",
    "gdp_usd":       "NY.GDP.MKTP.CD",
    "interest_rate": "FR.INR.LEND",
}

# ──────────────────────────────────────────────────────────────────────────────
# Demo Data  (curated 2024-2025 figures)
# ──────────────────────────────────────────────────────────────────────────────

_ECO_DEMO: list[dict] = [
    {"code": "AR", "name": "Argentina",      "gdp_usd": 0.64e12,  "gdp_growth": -1.6,  "inflation": 143.0, "unemployment":  7.7, "interest_rate": 40.0},
    {"code": "AU", "name": "Australia",      "gdp_usd": 1.72e12,  "gdp_growth":  1.5,  "inflation":   3.8, "unemployment":  4.1, "interest_rate":  7.5},
    {"code": "BR", "name": "Brazil",         "gdp_usd": 2.18e12,  "gdp_growth":  3.2,  "inflation":   4.8, "unemployment":  7.1, "interest_rate": 13.5},
    {"code": "CA", "name": "Canada",         "gdp_usd": 2.14e12,  "gdp_growth":  1.5,  "inflation":   2.7, "unemployment":  6.3, "interest_rate":  7.2},
    {"code": "CN", "name": "China",          "gdp_usd": 17.79e12, "gdp_growth":  5.0,  "inflation":   0.2, "unemployment":  5.1, "interest_rate":  3.6},
    {"code": "DE", "name": "Germany",        "gdp_usd": 4.45e12,  "gdp_growth": -0.3,  "inflation":   2.6, "unemployment":  3.4, "interest_rate":  4.5},
    {"code": "ES", "name": "Spain",          "gdp_usd": 1.58e12,  "gdp_growth":  3.2,  "inflation":   2.8, "unemployment": 11.7, "interest_rate":  4.5},
    {"code": "FR", "name": "France",         "gdp_usd": 3.03e12,  "gdp_growth":  1.1,  "inflation":   2.3, "unemployment":  7.3, "interest_rate":  4.5},
    {"code": "GB", "name": "United Kingdom", "gdp_usd": 3.09e12,  "gdp_growth":  1.1,  "inflation":   2.5, "unemployment":  4.5, "interest_rate":  7.5},
    {"code": "ID", "name": "Indonesia",      "gdp_usd": 1.37e12,  "gdp_growth":  5.0,  "inflation":   2.8, "unemployment":  5.0, "interest_rate":  9.5},
    {"code": "IN", "name": "India",          "gdp_usd": 3.57e12,  "gdp_growth":  8.2,  "inflation":   4.9, "unemployment":  7.9, "interest_rate": 10.5},
    {"code": "IT", "name": "Italy",          "gdp_usd": 2.26e12,  "gdp_growth":  0.7,  "inflation":   1.2, "unemployment":  6.1, "interest_rate":  4.5},
    {"code": "JP", "name": "Japan",          "gdp_usd": 4.21e12,  "gdp_growth": -0.1,  "inflation":   2.7, "unemployment":  2.5, "interest_rate":  1.5},
    {"code": "KR", "name": "South Korea",    "gdp_usd": 1.71e12,  "gdp_growth":  2.5,  "inflation":   2.3, "unemployment":  2.7, "interest_rate":  5.1},
    {"code": "MX", "name": "Mexico",         "gdp_usd": 1.79e12,  "gdp_growth":  1.5,  "inflation":   4.7, "unemployment":  2.9, "interest_rate": 14.5},
    {"code": "RU", "name": "Russia",         "gdp_usd": 2.24e12,  "gdp_growth":  3.6,  "inflation":   8.5, "unemployment":  2.4, "interest_rate": 19.0},
    {"code": "SA", "name": "Saudi Arabia",   "gdp_usd": 1.10e12,  "gdp_growth":  1.3,  "inflation":   1.6, "unemployment":  8.3, "interest_rate":  6.5},
    {"code": "TR", "name": "Turkey",         "gdp_usd": 1.11e12,  "gdp_growth":  3.2,  "inflation":  65.0, "unemployment":  8.5, "interest_rate": 42.0},
    {"code": "US", "name": "United States",  "gdp_usd": 27.36e12, "gdp_growth":  2.8,  "inflation":   2.9, "unemployment":  4.0, "interest_rate":  8.5},
    {"code": "ZA", "name": "South Africa",   "gdp_usd": 0.38e12,  "gdp_growth":  0.7,  "inflation":   4.7, "unemployment": 32.9, "interest_rate": 11.5},
]

_INDEX_DEMO: list[dict] = [
    {"ticker": "^GSPC",     "name": "S&P 500",        "country": "US", "value": 5_800.45,   "change_pct":  0.52},
    {"ticker": "^DJI",      "name": "Dow Jones",       "country": "US", "value": 42_008.20,  "change_pct":  0.31},
    {"ticker": "^IXIC",     "name": "NASDAQ",          "country": "US", "value": 18_542.60,  "change_pct":  0.73},
    {"ticker": "^FTSE",     "name": "FTSE 100",        "country": "GB", "value": 8_215.70,   "change_pct": -0.12},
    {"ticker": "^N225",     "name": "Nikkei 225",      "country": "JP", "value": 38_145.30,  "change_pct":  1.18},
    {"ticker": "^HSI",      "name": "Hang Seng",       "country": "HK", "value": 19_480.50,  "change_pct": -0.47},
    {"ticker": "^DAX",      "name": "DAX",             "country": "DE", "value": 18_032.80,  "change_pct":  0.28},
    {"ticker": "^CAC40",    "name": "CAC 40",          "country": "FR", "value":  7_612.40,  "change_pct": -0.21},
    {"ticker": "^AXJO",     "name": "ASX 200",         "country": "AU", "value":  8_098.60,  "change_pct":  0.38},
    {"ticker": "^BSESN",    "name": "BSE Sensex",      "country": "IN", "value": 73_215.90,  "change_pct":  0.84},
    {"ticker": "^KS11",     "name": "KOSPI",           "country": "KR", "value":  2_512.30,  "change_pct":  0.19},
    {"ticker": "^BVSP",     "name": "Bovespa",         "country": "BR", "value": 128_340.00, "change_pct": -0.33},
    {"ticker": "^STOXX50E", "name": "Euro Stoxx 50",   "country": "EU", "value":  4_912.70,  "change_pct":  0.14},
]

_FOREX_DEMO: list[dict] = [
    {"pair": "USD/EUR", "currency": "EUR", "rate": 0.9218},
    {"pair": "USD/GBP", "currency": "GBP", "rate": 0.7891},
    {"pair": "USD/JPY", "currency": "JPY", "rate": 154.82},
    {"pair": "USD/CNY", "currency": "CNY", "rate":  7.240},
    {"pair": "USD/INR", "currency": "INR", "rate": 84.52},
    {"pair": "USD/BRL", "currency": "BRL", "rate":  5.195},
    {"pair": "USD/CAD", "currency": "CAD", "rate":  1.362},
    {"pair": "USD/AUD", "currency": "AUD", "rate":  1.553},
    {"pair": "USD/KRW", "currency": "KRW", "rate": 1382.0},
    {"pair": "USD/MXN", "currency": "MXN", "rate": 17.08},
    {"pair": "USD/CHF", "currency": "CHF", "rate":  0.901},
    {"pair": "USD/SGD", "currency": "SGD", "rate":  1.348},
    {"pair": "USD/TRY", "currency": "TRY", "rate": 32.85},
    {"pair": "USD/ZAR", "currency": "ZAR", "rate": 18.54},
]


# ──────────────────────────────────────────────────────────────────────────────
# Live Data Fetchers
# ──────────────────────────────────────────────────────────────────────────────

class WorldBankFetcher:
    BASE_URL = "https://api.worldbank.org/v2"
    TIMEOUT = 15

    def fetch_indicator(self, country_code: str, indicator: str, lookback: int = 5) -> float | None:
        url = (
            f"{self.BASE_URL}/country/{country_code}/indicator/{indicator}"
            f"?format=json&mrv={lookback}&per_page={lookback}"
        )
        try:
            resp = requests.get(url, timeout=self.TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            if len(payload) < 2 or not payload[1]:
                return None
            for entry in payload[1]:
                if entry.get("value") is not None:
                    return round(float(entry["value"]), 2)
        except Exception:
            pass
        return None

    def fetch_country_data(self, country_code: str) -> dict:
        row: dict = {"code": country_code, "name": COUNTRIES.get(country_code, country_code)}
        for key, indicator in WORLD_BANK_INDICATORS.items():
            row[key] = self.fetch_indicator(country_code, indicator)
        return row

    def fetch_all(self, country_codes: list[str]) -> list[dict]:
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(self.fetch_country_data, c): c for c in country_codes}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    pass
        results.sort(key=lambda r: r["name"])
        return results


class MarketFetcher:
    TIMEOUT = 10

    def fetch_indices(self) -> list[dict]:
        try:
            import yfinance as yf
            tickers_str = " ".join(STOCK_INDICES.keys())
            data = yf.download(tickers_str, period="2d", interval="1d",
                               progress=False, auto_adjust=True, threads=True)
            close = data.get("Close")
            if close is None:
                return []
            results: list[dict] = []
            for ticker, (name, country) in STOCK_INDICES.items():
                try:
                    vals = close[ticker].dropna()
                    if not len(vals):
                        continue
                    current = float(vals.iloc[-1])
                    prev = float(vals.iloc[-2]) if len(vals) >= 2 else current
                    chg = (current - prev) / prev * 100 if prev else 0.0
                    results.append({"ticker": ticker, "name": name, "country": country,
                                    "value": current, "change_pct": round(chg, 2)})
                except Exception:
                    pass
            return sorted(results, key=lambda r: r["name"])
        except Exception:
            return []

    def fetch_forex(self) -> list[dict]:
        try:
            resp = requests.get(
                f"https://open.er-api.com/v6/latest/{BASE_FOREX_CURRENCY}",
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            rates = resp.json().get("rates", {})
            return [
                {"pair": f"{BASE_FOREX_CURRENCY}/{c}", "currency": c, "rate": round(float(rates[c]), 4)}
                for c in FOREX_TARGETS if c in rates
            ]
        except Exception:
            return []


# ──────────────────────────────────────────────────────────────────────────────
# Data Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_data(live: bool, country_filter: set[str] | None) -> tuple[list, list, list, bool]:
    """Return (eco_data, index_data, forex_data, is_live)."""
    if not live:
        eco = [r for r in _ECO_DEMO if not country_filter or r["code"] in country_filter]
        return eco, _INDEX_DEMO, _FOREX_DEMO, False

    eco: list = []
    indices: list = []
    forex: list = []

    def _eco():
        nonlocal eco
        codes = list(country_filter or COUNTRIES.keys())
        eco = WorldBankFetcher().fetch_all(codes)

    def _markets():
        nonlocal indices
        indices = MarketFetcher().fetch_indices()

    def _forex():
        nonlocal forex
        forex = MarketFetcher().fetch_forex()

    threads = [
        threading.Thread(target=_eco),
        threading.Thread(target=_markets),
        threading.Thread(target=_forex),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Fall back to demo where live fetch returned nothing
    if not eco:
        eco = [r for r in _ECO_DEMO if not country_filter or r["code"] in country_filter]
    if not indices:
        indices = _INDEX_DEMO
    if not forex:
        forex = _FOREX_DEMO

    return eco, indices, forex, True


# ──────────────────────────────────────────────────────────────────────────────
# Rendering Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pct(value: float | None, reverse: bool = False, suffix: str = "%") -> Text:
    if value is None:
        return Text("N/A", style="dim")
    positive_is_good = not reverse
    if value > 0:
        style = "green" if positive_is_good else "red"
        arrow = "▲"
    elif value < 0:
        style = "red" if positive_is_good else "green"
        arrow = "▼"
    else:
        style, arrow = "dim", " "
    return Text(f"{arrow} {abs(value):.2f}{suffix}", style=style)


def _gdp(value: float | None) -> Text:
    if value is None:
        return Text("N/A", style="dim")
    t = value / 1e12
    return Text(f"${t:.2f}T")


def _flag(country: str) -> str:
    flags = {
        "US": "🇺🇸", "CN": "🇨🇳", "JP": "🇯🇵", "DE": "🇩🇪", "GB": "🇬🇧",
        "IN": "🇮🇳", "FR": "🇫🇷", "BR": "🇧🇷", "CA": "🇨🇦", "AU": "🇦🇺",
        "KR": "🇰🇷", "MX": "🇲🇽", "ID": "🇮🇩", "SA": "🇸🇦", "ZA": "🇿🇦",
        "RU": "🇷🇺", "IT": "🇮🇹", "ES": "🇪🇸", "AR": "🇦🇷", "TR": "🇹🇷",
        "HK": "🇭🇰", "EU": "🇪🇺",
    }
    return flags.get(country, "  ")


# ──────────────────────────────────────────────────────────────────────────────
# Table Builders
# ──────────────────────────────────────────────────────────────────────────────

def make_eco_table(eco_data: list[dict]) -> Table:
    t = Table(
        title="[bold]Economic Indicators[/bold]  [dim](World Bank — latest available year)[/dim]",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_lines=True,
        title_style="white",
        min_width=88,
    )
    t.add_column("", width=2)                                 # flag
    t.add_column("Country",       style="bold white", min_width=16)
    t.add_column("GDP (USD)",     justify="right",    min_width=9)
    t.add_column("GDP Growth",    justify="right",    min_width=11)
    t.add_column("Inflation",     justify="right",    min_width=10)
    t.add_column("Unemployment",  justify="right",    min_width=12)
    t.add_column("Lending Rate",  justify="right",    min_width=12)

    for row in eco_data:
        t.add_row(
            _flag(row["code"]),
            row["name"],
            _gdp(row.get("gdp_usd")),
            _pct(row.get("gdp_growth")),
            _pct(row.get("inflation"),    reverse=True),
            _pct(row.get("unemployment"), reverse=True),
            _pct(row.get("interest_rate"), reverse=True),
        )
    return t


def make_index_table(index_data: list[dict]) -> Table:
    t = Table(
        title="[bold]Global Stock Market Indices[/bold]",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_lines=True,
        title_style="white",
        min_width=60,
    )
    t.add_column("",         width=2)
    t.add_column("Index",    style="bold white", min_width=16)
    t.add_column("Ticker",   style="dim",        min_width=10)
    t.add_column("Value",    justify="right",    min_width=14)
    t.add_column("Daily Δ",  justify="right",    min_width=10)

    for row in sorted(index_data, key=lambda r: r["name"]):
        t.add_row(
            _flag(row["country"]),
            row["name"],
            row["ticker"],
            f"{row['value']:>14,.2f}",
            _pct(row["change_pct"]),
        )
    return t


def make_forex_table(forex_data: list[dict]) -> Table:
    t = Table(
        title=f"[bold]Forex Rates[/bold]  [dim](Base: {BASE_FOREX_CURRENCY})[/dim]",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_lines=True,
        title_style="white",
        min_width=50,
    )
    # Two pairs per row
    t.add_column("Pair",   style="bold white", min_width=10)
    t.add_column("Rate",   justify="right",    min_width=12)
    t.add_column("Pair ",  style="bold white", min_width=10)
    t.add_column("Rate ",  justify="right",    min_width=12)

    half = (len(forex_data) + 1) // 2
    left, right = forex_data[:half], forex_data[half:]
    for i, l in enumerate(left):
        r = right[i] if i < len(right) else None
        t.add_row(
            l["pair"],  f"{l['rate']:>12.4f}",
            r["pair"] if r else "", f"{r['rate']:>12.4f}" if r else "",
        )
    return t


def make_header(is_live: bool, ts: str) -> Panel:
    source = "[green]LIVE[/green]" if is_live else "[yellow]DEMO DATA[/yellow] [dim](2024-2025)[/dim]"
    content = Align.center(Text.assemble(
        Text("Global Economies Monitor\n", style="bold white"),
        Text(f"Updated: {ts}  |  Source: "),
        Text(source),
    ))
    return Panel(content, style="bold blue", padding=(0, 4))


def make_summary_panel(eco_data: list[dict]) -> Panel:
    """Quick-glance stats: fastest growers, highest inflation, etc."""
    valid = [r for r in eco_data if r.get("gdp_growth") is not None]
    if not valid:
        return Panel("No data", title="Highlights")

    sorted_growth = sorted(valid, key=lambda r: r["gdp_growth"], reverse=True)
    sorted_inf    = sorted(
        [r for r in eco_data if r.get("inflation") is not None],
        key=lambda r: r["inflation"], reverse=True,
    )
    sorted_unemp  = sorted(
        [r for r in eco_data if r.get("unemployment") is not None],
        key=lambda r: r["unemployment"], reverse=True,
    )

    def top3(lst: list[dict], field: str) -> str:
        return ", ".join(f"{r['name']} ({r[field]:.1f}%)" for r in lst[:3])

    lines = [
        f"[green]Fastest growing:[/green]    {top3(sorted_growth, 'gdp_growth')}",
        f"[red]Slowest / contracting:[/red] {top3(sorted_growth[::-1], 'gdp_growth')}",
        f"[yellow]Highest inflation:[/yellow]  {top3(sorted_inf, 'inflation')}",
        f"[cyan]Lowest inflation:[/cyan]    {top3(sorted_inf[::-1], 'inflation')}",
        f"[red]Highest unemployment:[/red]  {top3(sorted_unemp, 'unemployment')}",
    ]
    return Panel("\n".join(lines), title="[bold]Highlights[/bold]", border_style="dim white")


# ──────────────────────────────────────────────────────────────────────────────
# Main Render
# ──────────────────────────────────────────────────────────────────────────────

def render(eco: list, indices: list, forex: list, is_live: bool) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    console.print(make_header(is_live, ts))
    console.print()

    if eco:
        console.print(make_summary_panel(eco))
        console.print()
        console.print(make_eco_table(eco))
        console.print()

    if indices:
        console.print(make_index_table(indices))
        console.print()

    if forex:
        console.print(make_forex_table(forex))
        console.print()

    console.print(Rule(style="dim"))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Global Economies Monitor — terminal dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monitor.py                        # Demo data (offline)
  python monitor.py --live                 # Fetch from World Bank, yfinance, FX API
  python monitor.py --live --refresh 300   # Auto-refresh every 5 minutes
  python monitor.py --countries US CN JP   # Filter to specific countries
""",
    )
    p.add_argument("--live", action="store_true",
                   help="Fetch real-time data from external APIs")
    p.add_argument("--refresh", type=int, default=0, metavar="SECONDS",
                   help="Auto-refresh interval in seconds (0 = one-shot)")
    p.add_argument("--countries", nargs="+", metavar="CODE",
                   help=f"ISO-2 country codes. Valid: {', '.join(sorted(COUNTRIES))}")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    country_filter: set[str] | None = None
    if args.countries:
        codes = {c.upper() for c in args.countries}
        invalid = codes - set(COUNTRIES)
        if invalid:
            console.print(f"[yellow]Unknown codes skipped:[/yellow] {', '.join(sorted(invalid))}")
        country_filter = codes & set(COUNTRIES)

    def run_once() -> None:
        eco, indices, forex, is_live = load_data(args.live, country_filter)
        console.clear()
        render(eco, indices, forex, is_live)

    if args.refresh > 0:
        console.print(f"[bold]Auto-refresh every {args.refresh}s. Press Ctrl+C to stop.[/bold]")
        try:
            while True:
                run_once()
                console.print(f"[dim]Next refresh in {args.refresh}s…[/dim]")
                time.sleep(args.refresh)
        except KeyboardInterrupt:
            console.print("\n[bold]Stopped.[/bold]")
    else:
        run_once()


if __name__ == "__main__":
    main()
