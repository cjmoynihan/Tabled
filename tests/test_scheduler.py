"""Test the by-game scheduler and resumability against a mock API."""
import math, sqlite3, sys
from xml.etree import ElementTree as ET
from bgg import bgg_collect as B

TOTALS = {101: 250, 102: 100, 103: 1000, 104: 40}  # ratings per game

class MockClient:
    def __init__(self, fail_after=None):
        self.calls, self.fail_after, self.request_count = [], fail_after, 0
    def get(self, endpoint, params, accept_202=False):
        self.calls.append((params["id"], params["page"]))
        if self.fail_after and len(self.calls) > self.fail_after:
            raise KeyboardInterrupt
        page = int(params["page"])
        parts = []
        for gid in (int(x) for x in str(params["id"]).split(",")):
            total = TOTALS[gid]
            start = (page - 1) * B.PAGE_SIZE
            n = max(0, min(B.PAGE_SIZE, total - start))
            comments = "".join(
                f'<comment username="u{start+i}" rating="{(i%10)+1}" value=""/>'
                for i in range(n))
            parts.append(f'<item type="boardgame" id="{gid}">'
                         f'<comments page="{page}" totalitems="{total}">{comments}'
                         f'</comments></item>')
        return ET.fromstring(f"<items>{''.join(parts)}</items>")

def fresh():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    B.init_db(conn)
    conn.executemany("INSERT INTO games (game_id, name, num_ratings) VALUES (?,?,?)",
                     [(g, f"G{g}", n) for g, n in TOTALS.items()])
    conn.commit()
    return conn

fails = []
def check(label, got, want):
    if got == want: print(f"  PASS  {label}")
    else: print(f"  FAIL  {label}: got {got!r}, want {want!r}"); fails.append(label)

expected_ratings = sum(TOTALS.values())
expected_pages = sum(math.ceil(n / B.PAGE_SIZE) for n in TOTALS.values())

print("Full run:")
conn, mc = fresh(), MockClient()
B.collect_by_game(conn, mc, top=None)
check("all ratings collected", conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0],
      expected_ratings)
check("all pages checkpointed", conn.execute("SELECT COUNT(*) FROM game_pages").fetchone()[0],
      expected_pages)
# 4 games need page 1, 2 need page 2, 1 needs pages 3-10 => batching packs page 1-2
check("batched into fewer requests than pages", len(mc.calls) < expected_pages, True)
check("page 1 batched all 4 games", mc.calls[0][0].count(",") + 1, 4)

print("\nInterrupt then resume:")
conn, mc1 = fresh(), MockClient(fail_after=3)
try: B.collect_by_game(conn, mc1, top=None)
except KeyboardInterrupt: pass
partial = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
check("partial progress persisted", 0 < partial < expected_ratings, True)

mc2 = MockClient()
B.collect_by_game(conn, mc2, top=None)
check("resume completes dataset",
      conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0], expected_ratings)
check("resume skipped finished pages", len(mc2.calls) < len(mc.calls), True)

print("\nRerun on complete DB:")
mc3 = MockClient()
B.collect_by_game(conn, mc3, top=None)
check("no wasted requests", len(mc3.calls), 0)

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILURES: ' + str(fails)}")
sys.exit(1 if fails else 0)