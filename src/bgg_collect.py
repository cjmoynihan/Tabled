#!/usr/bin/env python3
"""
collect.py — collect BoardGameGeek user ratings into SQLite.

Two collection strategies, both resumable:

  by-game   thing?id=...&ratingcomments=1  -> 100 ratings/page, up to 20 game ids
            per request. Complete columns of the user x game matrix.

  by-user   collection?username=...&rated=1 -> ALL of one user's ratings in a
            single request, but one username at a time and often HTTP 202
            (queued) first. Complete rows of the matrix.

Usage:
    python bgg_collect.py init
    python bgg_collect.py seed-games --csv games.csv
    python bgg_collect.py estimate
    python bgg_collect.py by-game --top 2000
    python bgg_collect.py discover-users
    python bgg_collect.py by-user --limit 500
    python bgg_collect.py stats

Rate limiting is global and serial on purpose: BGG throttles per-IP, so threads
do not increase throughput, they just get you 503s faster.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import logging
import math
import os
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from xml.etree import ElementTree as ET

import requests

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

API_ROOT = "https://boardgamegeek.com/xmlapi2"  # no www. — it breaks auth
DB_PATH = Path(os.environ.get("BGG_DB", "bgg.sqlite"))
CACHE_DIR = Path(os.environ.get("BGG_CACHE", "raw_cache"))

MIN_INTERVAL = float(os.environ.get("BGG_MIN_INTERVAL", "5.0"))  # seconds
PAGE_SIZE = 100          # API maximum
MAX_IDS_PER_THING = 20   # API maximum
MAX_RETRIES = 6
USER_AGENT = os.environ.get(
    "BGG_USER_AGENT",
    "bgg-recommender-research/0.1 (contact: you@example.com)",
)

log = logging.getLogger("bgg")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Throttled HTTP client
# --------------------------------------------------------------------------

class Throttle:
    """Serial rate limiter: guarantees >= min_interval between requests."""

    def __init__(self, min_interval: float = MIN_INTERVAL):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()


class BGGClient:
    """Handles throttling, retries, 202-queued responses, and raw caching."""

    def __init__(self, throttle: Throttle | None = None, cache: bool = True):
        self.throttle = throttle or Throttle()
        self.cache = cache
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/xml",
        })
        self.request_count = 0
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -- raw cache ---------------------------------------------------------

    def _cache_path(self, endpoint: str, params: dict) -> Path:
        key = endpoint + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        digest = hashlib.sha1(key.encode()).hexdigest()
        return CACHE_DIR / endpoint / digest[:2] / f"{digest}.xml.gz"

    def _cache_read(self, path: Path) -> bytes | None:
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rb") as fh:
                return fh.read()
        except OSError:
            log.warning("corrupt cache file, refetching: %s", path)
            return None

    def _cache_write(self, path: Path, body: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wb") as fh:
            fh.write(body)
        tmp.replace(path)

    # -- fetch -------------------------------------------------------------

    def get(self, endpoint: str, params: dict, accept_202: bool = False) -> ET.Element:
        """Fetch and parse an endpoint. Raises on unrecoverable failure."""
        cache_path = self._cache_path(endpoint, params)
        if self.cache:
            cached = self._cache_read(cache_path)
            if cached is not None:
                return ET.fromstring(cached)

        url = f"{API_ROOT}/{endpoint}"
        backoff = 5.0

        for attempt in range(1, MAX_RETRIES + 1):
            self.throttle.wait()
            try:
                resp = self.session.get(url, params=params, timeout=60)
            except requests.RequestException as exc:
                log.warning("network error (%s/%s): %s", attempt, MAX_RETRIES, exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue

            self.request_count += 1

            # 202: BGG has queued the request (collection endpoint mainly).
            if resp.status_code == 202:
                if not accept_202:
                    log.warning("unexpected 202 from %s", endpoint)
                wait = min(10.0 * attempt, 120.0)
                log.info("202 queued, waiting %.0fs (%s/%s)", wait, attempt, MAX_RETRIES)
                time.sleep(wait)
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                log.warning("HTTP %s, backing off %.0fs (%s/%s)",
                            resp.status_code, wait, attempt, MAX_RETRIES)
                time.sleep(wait)
                backoff = min(backoff * 2, 300)
                continue

            if resp.status_code == 404:
                raise LookupError(f"404 for {endpoint} {params}")

            resp.raise_for_status()

            body = resp.content
            try:
                root = ET.fromstring(body)
            except ET.ParseError as exc:
                log.warning("bad XML (%s/%s): %s", attempt, MAX_RETRIES, exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue

            # BGG returns 200 with an <errors> body for bad usernames etc.
            if root.tag == "errors":
                msg = root.findtext(".//message") or "unknown API error"
                raise LookupError(msg)

            if self.cache:
                self._cache_write(cache_path, body)
            return root

        raise RuntimeError(f"giving up on {endpoint} {params} after {MAX_RETRIES} tries")


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS games (
    game_id      INTEGER PRIMARY KEY,
    name         TEXT,
    year         INTEGER,
    num_ratings  INTEGER,          -- from seed CSV; refreshed by by-game
    avg_rating   REAL,
    bgg_rank     INTEGER
);

CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    first_seen    TEXT NOT NULL,
    fetched_at    TEXT,             -- when their collection was pulled
    fetch_status  TEXT,             -- ok | invalid | error
    n_ratings     INTEGER
);

CREATE TABLE IF NOT EXISTS ratings (
    user_id    INTEGER NOT NULL REFERENCES users(user_id),
    game_id    INTEGER NOT NULL,
    rating     REAL    NOT NULL,
    source     TEXT    NOT NULL,    -- 'game' | 'user'
    fetched_at TEXT    NOT NULL,
    PRIMARY KEY (user_id, game_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS ix_ratings_game ON ratings(game_id);

-- checkpoint table: which (game, page) pairs are already stored
CREATE TABLE IF NOT EXISTS game_pages (
    game_id     INTEGER NOT NULL,
    page        INTEGER NOT NULL,
    total_items INTEGER,
    n_returned  INTEGER,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (game_id, page)
) WITHOUT ROWID;
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
    log.info("schema ready at %s", DB_PATH)


def upsert_users(conn: sqlite3.Connection, usernames: Iterable[str]) -> dict[str, int]:
    """Insert usernames if new; return {username: user_id} for all of them."""
    names = sorted({u.strip() for u in usernames if u and u.strip()})
    if not names:
        return {}
    now = utcnow()
    conn.executemany(
        "INSERT OR IGNORE INTO users (username, first_seen) VALUES (?, ?)",
        [(n, now) for n in names],
    )
    out: dict[str, int] = {}
    for chunk in batched(names, 500):
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT user_id, username FROM users WHERE username IN ({marks})",
            chunk,
        ).fetchall()
        for r in rows:
            out[r["username"].lower()] = r["user_id"]
    return out


def insert_ratings(
    conn: sqlite3.Connection,
    rows: Sequence[tuple[int, int, float]],
    source: str,
) -> None:
    """Upsert ratings. A 'user'-sourced rating overwrites a 'game'-sourced one."""
    now = utcnow()
    conn.executemany(
        """
        INSERT INTO ratings (user_id, game_id, rating, source, fetched_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id, game_id) DO UPDATE SET
            rating     = excluded.rating,
            source     = excluded.source,
            fetched_at = excluded.fetched_at
        WHERE excluded.source = 'user' OR ratings.source = excluded.source
        """,
        [(u, g, r, source, now) for u, g, r in rows],
    )


def batched(seq: Sequence, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield list(seq[i:i + n])


# --------------------------------------------------------------------------
# Seeding the game list
# --------------------------------------------------------------------------

def seed_games_from_csv(conn: sqlite3.Connection, path: Path) -> int:
    """
    Load games from a CSV. Tries common column names from the BGG rank dump
    and from typical Kaggle board game datasets.
    """
    id_keys     = ("id", "game_id", "objectid", "bgg_id", "gameid")
    name_keys   = ("name", "primary", "game_name", "title")
    count_keys  = ("usersrated", "users_rated", "num_ratings", "numratings",
                   "ratings_count", "owned")
    avg_keys    = ("average", "avg_rating", "averageweight", "rating_average",
                   "bayesaverage")
    year_keys   = ("yearpublished", "year", "year_published")
    rank_keys   = ("rank", "bgg_rank", "boardgamerank")

    def pick(row: dict, keys: tuple[str, ...]) -> str | None:
        lowered = {k.lower().strip(): v for k, v in row.items() if k}
        for k in keys:
            if k in lowered and lowered[k] not in ("", None):
                return lowered[k]
        return None

    def as_int(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    def as_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gid = as_int(pick(row, id_keys))
            if gid is None:
                continue
            rows.append((
                gid,
                pick(row, name_keys),
                as_int(pick(row, year_keys)),
                as_int(pick(row, count_keys)),
                as_float(pick(row, avg_keys)),
                as_int(pick(row, rank_keys)),
            ))

    if not rows:
        raise SystemExit(
            f"No game ids found in {path}. Columns present: check that one of "
            f"{id_keys} exists."
        )

    conn.executemany(
        """
        INSERT INTO games (game_id, name, year, num_ratings, avg_rating, bgg_rank)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_id) DO UPDATE SET
            name        = COALESCE(excluded.name, games.name),
            year        = COALESCE(excluded.year, games.year),
            num_ratings = COALESCE(excluded.num_ratings, games.num_ratings),
            avg_rating  = COALESCE(excluded.avg_rating, games.avg_rating),
            bgg_rank    = COALESCE(excluded.bgg_rank, games.bgg_rank)
        """,
        rows,
    )
    conn.commit()
    log.info("seeded %d games from %s", len(rows), path)
    return len(rows)


# --------------------------------------------------------------------------
# Strategy A: by game (thing?ratingcomments=1)
# --------------------------------------------------------------------------

def parse_rating_comments(root: ET.Element) -> dict[int, dict]:
    """
    Parse a thing response into {game_id: {"total": int, "ratings": [(user, val)]}}.
    Works for single or batched id requests.
    """
    out: dict[int, dict] = {}
    for item in root.findall("item"):
        try:
            gid = int(item.attrib["id"])
        except (KeyError, ValueError):
            continue
        comments = item.find("comments")
        entry = {"total": 0, "ratings": []}
        if comments is not None:
            entry["total"] = int(comments.attrib.get("totalitems", 0) or 0)
            for c in comments.findall("comment"):
                username = c.attrib.get("username")
                raw = c.attrib.get("rating")
                if not username or raw in (None, "", "N/A"):
                    continue
                try:
                    val = float(raw)
                except ValueError:
                    continue
                if 1.0 <= val <= 10.0:
                    entry["ratings"].append((username, val))
        out[gid] = entry
    return out


def store_page(
    conn: sqlite3.Connection,
    game_id: int,
    page: int,
    parsed: dict,
) -> int:
    """Write one (game, page) worth of ratings plus its checkpoint row."""
    pairs = parsed["ratings"]
    uid_map = upsert_users(conn, (u for u, _ in pairs))
    rows = []
    for username, val in pairs:
        uid = uid_map.get(username.lower())
        if uid is not None:
            rows.append((uid, game_id, val))
    if rows:
        insert_ratings(conn, rows, source="game")
    conn.execute(
        """
        INSERT INTO game_pages (game_id, page, total_items, n_returned, fetched_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(game_id, page) DO UPDATE SET
            total_items = excluded.total_items,
            n_returned  = excluded.n_returned,
            fetched_at  = excluded.fetched_at
        """,
        (game_id, page, parsed["total"], len(pairs), utcnow()),
    )
    return len(rows)


def target_games(conn: sqlite3.Connection, top: int | None) -> list[sqlite3.Row]:
    sql = """
        SELECT game_id, name, num_ratings
        FROM games
        WHERE num_ratings IS NOT NULL AND num_ratings > 0
        ORDER BY num_ratings DESC
    """
    if top:
        sql += " LIMIT ?"
        return conn.execute(sql, (top,)).fetchall()
    return conn.execute(sql).fetchall()


def pages_needed(conn: sqlite3.Connection, games: Sequence[sqlite3.Row]) -> dict[int, int]:
    """
    How many pages each game needs. Prefers the authoritative totalitems we
    saw from a previous fetch; falls back to the seed CSV's rating count.
    """
    known = {
        r["game_id"]: r["total_items"]
        for r in conn.execute(
            "SELECT game_id, MAX(total_items) AS total_items FROM game_pages "
            "GROUP BY game_id"
        )
        if r["total_items"]
    }
    out = {}
    for g in games:
        total = known.get(g["game_id"]) or g["num_ratings"] or 0
        out[g["game_id"]] = max(1, math.ceil(total / PAGE_SIZE))
    return out


def collect_by_game(
    conn: sqlite3.Connection,
    client: BGGClient,
    top: int | None = None,
    max_pages_per_game: int | None = None,
) -> None:
    """
    Batched, resumable collection.

    The API takes up to 20 ids per request but only ONE page number, so the
    work queue is grouped by page: page 1 packs 20 games per request, deep
    pages only pack the few games that have that many ratings.
    """
    games = target_games(conn, top)
    if not games:
        raise SystemExit("No games with rating counts. Run seed-games first.")

    done = {
        (r["game_id"], r["page"])
        for r in conn.execute("SELECT game_id, page FROM game_pages")
    }
    needed = pages_needed(conn, games)

    # Group outstanding work by page number.
    by_page: dict[int, list[int]] = {}
    for gid, n_pages in needed.items():
        if max_pages_per_game:
            n_pages = min(n_pages, max_pages_per_game)
        for page in range(1, n_pages + 1):
            if (gid, page) not in done:
                by_page.setdefault(page, []).append(gid)

    total_requests = sum(math.ceil(len(v) / MAX_IDS_PER_THING) for v in by_page.values())
    log.info(
        "%d games, %d (game,page) tasks outstanding -> ~%d requests, "
        "~%.1f h at %.1fs/request",
        len(games),
        sum(len(v) for v in by_page.values()),
        total_requests,
        total_requests * MIN_INTERVAL / 3600,
        MIN_INTERVAL,
    )

    stored = 0
    done_requests = 0
    started = time.monotonic()

    for page in sorted(by_page):
        for chunk in batched(by_page[page], MAX_IDS_PER_THING):
            params = {
                "id": ",".join(str(g) for g in chunk),
                "ratingcomments": 1,
                "page": page,
                "pagesize": PAGE_SIZE,
            }
            try:
                root = client.get("thing", params)
            except LookupError as exc:
                log.warning("skipping page %s of %s: %s", page, chunk, exc)
                continue

            parsed = parse_rating_comments(root)
            if len(parsed) < len(chunk):
                missing = set(chunk) - set(parsed)
                log.warning("no data returned for ids %s", sorted(missing))

            for gid, entry in parsed.items():
                stored += store_page(conn, gid, page, entry)
            conn.commit()

            done_requests += 1
            if done_requests % 20 == 0:
                elapsed = time.monotonic() - started
                rate = stored / elapsed if elapsed else 0
                log.info(
                    "page %d | %d/%d requests | %d ratings stored | %.1f ratings/s",
                    page, done_requests, total_requests, stored, rate,
                )

    log.info("by-game finished: %d ratings stored in %d requests",
             stored, done_requests)


# --------------------------------------------------------------------------
# Strategy B: by user (collection?rated=1)
# --------------------------------------------------------------------------

def parse_collection(root: ET.Element) -> list[tuple[int, float]]:
    out = []
    for item in root.findall("item"):
        try:
            gid = int(item.attrib["objectid"])
        except (KeyError, ValueError):
            continue
        node = item.find("./stats/rating")
        if node is None:
            node = item.find(".//rating")
        if node is None:
            continue
        raw = node.attrib.get("value")
        if raw in (None, "", "N/A"):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        if 1.0 <= val <= 10.0:
            out.append((gid, val))
    return out


def fetch_one_collection(
    conn: sqlite3.Connection,
    client: BGGClient,
    user_id: int,
    username: str,
    restrict_to_games: bool = False,
) -> int:
    params = {
        "username": username,
        "rated": 1,
        "stats": 1,
        "excludesubtype": "boardgameexpansion",
    }
    try:
        root = client.get("collection", params, accept_202=True)
    except LookupError as exc:
        log.info("user %s unavailable: %s", username, exc)
        conn.execute(
            "UPDATE users SET fetched_at = ?, fetch_status = 'invalid' WHERE user_id = ?",
            (utcnow(), user_id),
        )
        conn.commit()
        return 0
    except RuntimeError as exc:
        log.warning("user %s failed: %s", username, exc)
        conn.execute(
            "UPDATE users SET fetched_at = ?, fetch_status = 'error' WHERE user_id = ?",
            (utcnow(), user_id),
        )
        conn.commit()
        return 0

    pairs = parse_collection(root)

    if restrict_to_games:
        known = {r[0] for r in conn.execute("SELECT game_id FROM games")}
        pairs = [(g, v) for g, v in pairs if g in known]

    if pairs:
        insert_ratings(conn, [(user_id, g, v) for g, v in pairs], source="user")
    conn.execute(
        "UPDATE users SET fetched_at = ?, fetch_status = 'ok', n_ratings = ? "
        "WHERE user_id = ?",
        (utcnow(), len(pairs), user_id),
    )
    conn.commit()
    return len(pairs)


def collect_by_user(
    conn: sqlite3.Connection,
    client: BGGClient,
    limit: int | None = None,
    sample: bool = False,
    restrict_to_games: bool = False,
) -> None:
    sql = "SELECT user_id, username FROM users WHERE fetched_at IS NULL"
    sql += " ORDER BY RANDOM()" if sample else " ORDER BY user_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    todo = conn.execute(sql).fetchall()

    if not todo:
        raise SystemExit("No unfetched users. Run discover-users or by-game first.")

    log.info("%d users to fetch -> ~%.1f h at %.1fs/request (plus 202 retries)",
             len(todo), len(todo) * MIN_INTERVAL / 3600, MIN_INTERVAL)

    total = 0
    for i, row in enumerate(todo, 1):
        total += fetch_one_collection(
            conn, client, row["user_id"], row["username"], restrict_to_games
        )
        if i % 25 == 0:
            log.info("%d/%d users | %d ratings | %.1f ratings/user",
                     i, len(todo), total, total / i)

    log.info("by-user finished: %d users, %d ratings, %.1f ratings/user",
             len(todo), total, total / max(1, len(todo)))


def discover_users(conn: sqlite3.Connection, client: BGGClient,
                   top: int | None = None, pages: int = 3) -> None:
    """
    Cheap username harvest: first N pages of ratings for the most-rated games.
    Enough to build a user list without full coverage of any game.
    """
    collect_by_game(conn, client, top=top, max_pages_per_game=pages)
    n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    log.info("user table now holds %d usernames", n)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def verify_batching(conn: sqlite3.Connection, client: BGGClient) -> None:
    """
    The whole runtime advantage of by-game rests on one assumption: that
    `id=a,b,c...&page=N` returns page N for EVERY id, not just the first, and
    not page 1 for all of them. Costs ~4 requests to confirm. Run this first.
    """
    games = conn.execute(
        "SELECT game_id, name FROM games WHERE num_ratings > 30000 "
        "ORDER BY num_ratings DESC LIMIT 3"
    ).fetchall()
    if len(games) < 2:
        raise SystemExit("Need >=2 games with >30k ratings seeded to verify.")

    ids = [g["game_id"] for g in games]
    probe_page = 5
    ok = True

    # 1. Batched request at a deep page.
    batched_resp = parse_rating_comments(
        client.get("thing", {"id": ",".join(map(str, ids)),
                             "ratingcomments": 1, "page": probe_page,
                             "pagesize": PAGE_SIZE})
    )
    print(f"\nBatched request, {len(ids)} ids, page {probe_page}:")
    for gid in ids:
        entry = batched_resp.get(gid)
        n = len(entry["ratings"]) if entry else 0
        print(f"  game {gid}: {n:>3} ratings, totalitems={entry['total'] if entry else '-'}")
        if n == 0:
            ok = False

    if len(batched_resp) < len(ids):
        print("  FAIL: not every id came back. Set MAX_IDS_PER_THING = 1.")
        ok = False

    # 2. Same game+page fetched alone — should match the batched slice.
    gid = ids[0]
    single = parse_rating_comments(
        client.get("thing", {"id": gid, "ratingcomments": 1,
                             "page": probe_page, "pagesize": PAGE_SIZE})
    )
    a = set(batched_resp.get(gid, {}).get("ratings", []))
    b = set(single.get(gid, {}).get("ratings", []))
    overlap = len(a & b) / max(1, len(b))
    print(f"\nBatched vs single for game {gid}, page {probe_page}: "
          f"{overlap:.0%} identical")
    if overlap < 0.9:
        print("  FAIL: batched response ignores the page param per-item.")
        ok = False

    # 3. Page 1 vs page 5 must differ, or paging is being ignored entirely.
    page1 = parse_rating_comments(
        client.get("thing", {"id": gid, "ratingcomments": 1,
                             "page": 1, "pagesize": PAGE_SIZE})
    )
    c = set(page1.get(gid, {}).get("ratings", []))
    if c and b and len(c & b) / len(b) > 0.5:
        print(f"\nFAIL: page 1 and page {probe_page} returned the same users.")
        ok = False
    else:
        print(f"Page 1 and page {probe_page} are distinct: OK")

    print(f"\n=> batching {'CONFIRMED, use --top with defaults' if ok else 'BROKEN, see failures above'}\n")


# --------------------------------------------------------------------------
# Cost estimation
# --------------------------------------------------------------------------

def estimate(conn: sqlite3.Connection, top: int | None,
             avg_per_user: float, retry_factor: float, coverage: float) -> None:
    games = target_games(conn, top)
    if not games:
        raise SystemExit("No games with rating counts. Run seed-games first.")

    counts = sorted((g["num_ratings"] or 0 for g in games), reverse=True)
    total_ratings = sum(counts)
    page_counts = sorted((max(1, math.ceil(c / PAGE_SIZE)) for c in counts), reverse=True)
    max_page = page_counts[0]

    # Requests for by-game: the API accepts 20 ids but only one page number, so
    # for each page p we pack the games that still need page p, 20 per request.
    # page_counts is sorted descending, so games needing page p are a prefix;
    # its length is found by binary search.
    requests_game = 0
    for p in range(1, max_page + 1):
        lo, hi = 0, len(page_counts)
        while lo < hi:
            mid = (lo + hi) // 2
            if page_counts[mid] >= p:
                lo = mid + 1
            else:
                hi = mid
        if lo == 0:
            break
        requests_game += math.ceil(lo / MAX_IDS_PER_THING)

    requests_game_unbatched = sum(page_counts)

    # Requests for by-user: how many users to cover the same rating volume.
    effective_per_user = avg_per_user * coverage
    users_needed = total_ratings / effective_per_user if effective_per_user else float("inf")
    requests_user = users_needed * retry_factor

    def hours(n):
        return n * MIN_INTERVAL / 3600

    print(f"\nScope: {len(games):,} games, {total_ratings:,} ratings to collect")
    print(f"Rate limit: {MIN_INTERVAL:.1f}s/request, page size {PAGE_SIZE}, "
          f"{MAX_IDS_PER_THING} ids/request\n")
    print(f"{'strategy':<34}{'requests':>12}{'hours':>10}{'days':>8}")
    print("-" * 64)
    for label, n in [
        ("by-game, 20 ids batched", requests_game),
        ("by-game, 1 id per request", requests_game_unbatched),
        (f"by-user ({avg_per_user:.0f} ratings/user)", requests_user),
    ]:
        print(f"{label:<34}{n:>12,.0f}{hours(n):>10,.1f}{hours(n)/24:>8,.1f}")
    print()
    print(f"Batching speedup: {requests_game_unbatched / max(1, requests_game):.1f}x")
    print(f"by-user assumes {coverage:.0%} of a user's ratings fall inside your "
          f"game set and {retry_factor:.1f} requests/user from 202 retries.")
    print("Measure the real numbers with: by-user --limit 50 --sample\n")


def stats(conn: sqlite3.Connection) -> None:
    def q(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    rows = [
        ("games seeded",       q("SELECT COUNT(*) FROM games")),
        ("users known",        q("SELECT COUNT(*) FROM users")),
        ("users fetched",      q("SELECT COUNT(*) FROM users WHERE fetch_status = 'ok'")),
        ("ratings total",      q("SELECT COUNT(*) FROM ratings")),
        ("  from game pages",  q("SELECT COUNT(*) FROM ratings WHERE source = 'game'")),
        ("  from collections", q("SELECT COUNT(*) FROM ratings WHERE source = 'user'")),
        ("game-pages done",    q("SELECT COUNT(*) FROM game_pages")),
    ]
    print()
    for label, value in rows:
        print(f"{label:<20}{value:>12,}")

    n_users, n_games, n_ratings = conn.execute(
        "SELECT COUNT(DISTINCT user_id), COUNT(DISTINCT game_id), COUNT(*) FROM ratings"
    ).fetchone()
    if n_users and n_games:
        density = n_ratings / (n_users * n_games)
        print(f"{'matrix density':<20}{density:>12.5%}")
    print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    global MIN_INTERVAL, DB_PATH

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--interval", type=float, default=MIN_INTERVAL,
                   help="seconds between requests (default %(default)s)")
    p.add_argument("--no-cache", action="store_true", help="skip raw XML cache")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    sub.add_parser("stats")
    sub.add_parser("verify-batching")

    sp = sub.add_parser("seed-games")
    sp.add_argument("--csv", type=Path, required=True)

    sp = sub.add_parser("by-game")
    sp.add_argument("--top", type=int, default=None)
    sp.add_argument("--max-pages", type=int, default=None,
                    help="cap pages per game (biases the sample — see notes)")

    sp = sub.add_parser("discover-users")
    sp.add_argument("--top", type=int, default=1000)
    sp.add_argument("--pages", type=int, default=3)

    sp = sub.add_parser("by-user")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--sample", action="store_true", help="random order")
    sp.add_argument("--restrict-to-games", action="store_true",
                    help="drop ratings for games not in the games table")

    sp = sub.add_parser("estimate")
    sp.add_argument("--top", type=int, default=None)
    sp.add_argument("--avg-per-user", type=float, default=120.0)
    sp.add_argument("--retry-factor", type=float, default=1.7)
    sp.add_argument("--coverage", type=float, default=0.7)

    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    MIN_INTERVAL = args.interval
    DB_PATH = args.db

    conn = connect(args.db)
    init_db(conn)
    client = BGGClient(Throttle(args.interval), cache=not args.no_cache)

    try:
        if args.cmd == "init":
            pass
        elif args.cmd == "stats":
            stats(conn)
        elif args.cmd == "verify-batching":
            verify_batching(conn, client)
        elif args.cmd == "seed-games":
            seed_games_from_csv(conn, args.csv)
        elif args.cmd == "by-game":
            collect_by_game(conn, client, args.top, args.max_pages)
        elif args.cmd == "discover-users":
            discover_users(conn, client, args.top, args.pages)
        elif args.cmd == "by-user":
            collect_by_user(conn, client, args.limit, args.sample,
                            args.restrict_to_games)
        elif args.cmd == "estimate":
            estimate(conn, args.top, args.avg_per_user,
                     args.retry_factor, args.coverage)
    except KeyboardInterrupt:
        log.info("interrupted — progress is committed, rerun to resume")
        return 130
    finally:
        conn.commit()
        conn.close()
        log.info("%d HTTP requests this run", client.request_count)

    return 0


if __name__ == "__main__":
    sys.exit(main())