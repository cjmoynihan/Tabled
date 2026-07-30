"""Offline tests: XML parsing + DB round trip, no network."""
import sqlite3, sys
from xml.etree import ElementTree as ET
from src import bgg_collect as B

THING = """<items>
  <item type="boardgame" id="13">
    <name type="primary" value="Catan"/>
    <comments page="5" totalitems="98765">
      <comment username="alice" rating="8" value="fun"/>
      <comment username="bob"   rating="10" value=""/>
      <comment username="carol" rating="N/A" value="no rating"/>
      <comment username=""      rating="7" value="anon"/>
      <comment username="dave"  rating="6.5" value=""/>
    </comments>
  </item>
  <item type="boardgame" id="822">
    <name type="primary" value="Carcassonne"/>
    <comments page="5" totalitems="120000">
      <comment username="alice" rating="9" value=""/>
      <comment username="erin"  rating="4" value=""/>
    </comments>
  </item>
  <item type="boardgame" id="999"/>
</items>"""

COLLECTION = """<items totalitems="3">
  <item objecttype="thing" objectid="13" subtype="boardgame">
    <name>Catan</name>
    <stats minplayers="3"><rating value="8"><average value="7.1"/></rating></stats>
  </item>
  <item objecttype="thing" objectid="174430" subtype="boardgame">
    <name>Gloomhaven</name>
    <stats><rating value="N/A"/></stats>
  </item>
  <item objecttype="thing" objectid="167791" subtype="boardgame">
    <name>Terraforming Mars</name>
    <stats><rating value="9.5"/></stats>
  </item>
</items>"""

fails = []
def check(label, got, want):
    if got == want:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}: got {got!r}, want {want!r}")
        fails.append(label)

print("parse_rating_comments:")
p = B.parse_rating_comments(ET.fromstring(THING))
check("games parsed", sorted(p), [13, 822, 999])
check("catan totalitems", p[13]["total"], 98765)
check("N/A + blank username dropped", sorted(u for u, _ in p[13]["ratings"]),
      ["alice", "bob", "dave"])
check("decimal rating kept", dict(p[13]["ratings"])["dave"], 6.5)
check("empty item -> no ratings", p[999]["ratings"], [])

print("\nparse_collection:")
c = B.parse_collection(ET.fromstring(COLLECTION))
check("N/A dropped, 2 kept", sorted(c), [(13, 8.0), (167791, 9.5)])

print("\nDB round trip:")
conn = sqlite3.connect(":memory:")
conn.row_factory = sqlite3.Row
B.init_db(conn)
conn.execute("INSERT INTO games (game_id, name, num_ratings) VALUES (13,'Catan',98765)")
for gid, entry in p.items():
    B.store_page(conn, gid, 5, entry)
conn.commit()

check("dedup: alice is one user", conn.execute(
    "SELECT COUNT(*) FROM users WHERE username='alice'").fetchone()[0], 1)
check("ratings stored", conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0], 5)
check("checkpoint rows", conn.execute("SELECT COUNT(*) FROM game_pages").fetchone()[0], 3)

# Idempotency: replaying the same page must not duplicate or corrupt.
for gid, entry in p.items():
    B.store_page(conn, gid, 5, entry)
conn.commit()
check("replay is idempotent", conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0], 5)

# Case-insensitive username matching (BGG usernames are case-preserving).
uids = B.upsert_users(conn, ["ALICE", "Alice", "alice"])
check("case-insensitive user match",
      conn.execute("SELECT COUNT(*) FROM users WHERE username LIKE 'alice'").fetchone()[0], 1)

# A 'user'-sourced rating should overwrite a 'game'-sourced one.
alice = conn.execute("SELECT user_id FROM users WHERE username='alice'").fetchone()[0]
B.insert_ratings(conn, [(alice, 13, 8.5)], source="user")
conn.commit()
row = conn.execute("SELECT rating, source FROM ratings WHERE user_id=? AND game_id=13",
                   (alice,)).fetchone()
check("collection overwrites game-page", (row["rating"], row["source"]), (8.5, "user"))

# ...but a game page must not clobber the authoritative collection value.
B.insert_ratings(conn, [(alice, 13, 8.0)], source="game")
conn.commit()
row = conn.execute("SELECT rating, source FROM ratings WHERE user_id=? AND game_id=13",
                   (alice,)).fetchone()
check("game-page does not clobber collection", (row["rating"], row["source"]), (8.5, "user"))

print(f"\n{'ALL PASS' if not fails else str(len(fails)) + ' FAILURES: ' + str(fails)}")
sys.exit(1 if fails else 0)