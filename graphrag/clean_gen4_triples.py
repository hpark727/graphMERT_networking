"""
Clean gen4 validated_both triples according to the gen4_audit.md recommendations.

Input:  gen4_triplets/validated_both/validated_ch{N}.csv
Output: gen4_triplets/filtered_and_validated/validated_ch{N}.csv

Filters applied (in order):
  1. Normalize entity names: underscores→spaces, unicode dashes→hyphen, lowercase
  2. Exact-duplicate removal (within chapter, after normalization)
  3. Trivial is_a tails (concept / metric / service / attack / protocol / ...)
  4. Instance-specific diagram entities (host a, router r1, node y, r1, s1, ...)
  5. Factually wrong or misleading triples
  6. Alice/Bob fictional characters
  7. Commercial product names (Netflix, YouTube, Gmail, ...)
  8. Explicit near-duplicate removal
  9. Conditional near-duplicate removal (weaker tail superseded by stronger)
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
IN_DIR  = _REPO / "gen4_triplets/validated_both"
OUT_DIR = _REPO / "gen4_triplets/filtered_and_validated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── 1. Normalization ────────────────────────────────────────────────────────

_UNICODE_DASHES = str.maketrans({
    '‐': '-',  # hyphen
    '‑': '-',  # non-breaking hyphen
    '‒': '-',  # figure dash
    '–': '-',  # en dash
    '—': '-',  # em dash
})

def normalize(text: str) -> str:
    return text.strip().lower().replace('_', ' ').translate(_UNICODE_DASHES)

# Entity-specific renames applied after general normalization.
# Keys and values are already-normalized strings.
_ENTITY_RENAME = {
    'bit torrent': 'bittorrent',
    'unreliable edge to edge message delivery service': 'unreliable channel',
}

def rename_entity(e: str) -> str:
    return _ENTITY_RENAME.get(e, e)


# ─── 2. Trivial is_a tails ───────────────────────────────────────────────────

TRIVIAL_ISA_TAILS = {
    'concept', 'metric', 'service', 'attack', 'protocol', 'algorithm',
    'pdu', 'channel', 'checksum', 'delay', 'packet', 'shared medium',
}

def is_trivial_isa(h: str, r: str, t: str) -> bool:
    return r == 'is_a' and t in TRIVIAL_ISA_TAILS


# ─── 3. Instance-specific entity detection ──────────────────────────────────

_INST_RE = re.compile(
    r'^('
    r'host [a-z]'                 # host a, host b, host c, host d, host e
    r'|router [a-z](?!\w)'        # router a, router b  (not "router r1" via next alt)
    r'|router r\d+'               # router r1, router r2
    r'|router \d+[a-z]+'          # router 1b, router 2c, router 3a, router 3d
    r'|r\d+'                      # r1, r2, r3, r5, r6  (standalone)
    r'|s\d+'                      # s1, s2  (standalone)
    r'|h\d+'                      # h1, h2  (standalone)
    r'|ap\d+'                     # ap2
    r'|node [a-z0-9](?!\w)'       # node 0..9, node a..z  (single char; not "node ntp")
    r'|station h\d+'              # station h1, station h2
    r'|switch s\d+'               # switch s1
    r'|destination [a-z]'         # destination a, destination d
    r'|prefix [a-z](?!\w)'        # prefix x  (single-letter math variable in BGP)
    r'|prefix \d[\d.]+/\d+'       # prefix 128.119.40.128/26
    r'|subnet \d+'                # subnet 1, subnet 2, subnet 3
    r'|address space [\d.]+/\d+'  # address space 10.0.0.0/24
    r'|[a-z]-[a-z] traffic'       # a-c traffic, b-d traffic
    r')$'
)

_INST_NAMES = {
    'salesperson laptop',
    'headquarters gateway router',
    'branch-office gateway router',
    'cisco 6500 router',
    'bob laptop',
}

def is_instance(entity: str) -> bool:
    return bool(_INST_RE.match(entity)) or entity in _INST_NAMES


# ─── 4. Factually wrong / misleading triples ────────────────────────────────

_WRONG = {
    ('traceroute',              'runs_on',   'source host'),       # wrong relation; tool ≠ process
    ('router',                  'causes',    'packet loss'),        # output buffer causes it; router is carrier
    ('packet',                  'increases', 'delay'),              # packets don't cause delay; congestion does
    ('n',                       'increases', 'distribution time'),  # n is a math variable, not a concept
    ('three-way tcp handshake', 'mitigates', 'playback attack'),    # non-standard term; should be replay attack
    ('tcp fin segment',         'enables',   'truncation attack'),  # fin doesn't enable; truncation exploits it
    ('hit rate',                'measures',  'web cache'),          # inverted: cache has hit rate, not vice versa
    ('source host',             'measures',  'round-trip delay'),   # traceroute measures, not the host itself
    ('source host',             'measures',  'round-trip time'),    # same issue; ch5 variant
    ('central ids processor',   'receives',  'ids sensor'),         # receives data from sensor, not sensor itself
    ('packet switching',        'enables',   'secure voice'),       # false; voice over PS requires extra security
}


# ─── 5. Alice/Bob fictional characters ──────────────────────────────────────

_ALICE_BOB = {'alice', 'bob'}
_ALICE_BOB_EXACT = {
    ('malicious host', 'sends', 'spam e-mail'),  # normalized form of the e‑mail triple
}

def is_alice_bob(h: str, r: str, t: str) -> bool:
    return h in _ALICE_BOB or t in _ALICE_BOB or (h, r, t) in _ALICE_BOB_EXACT


# ─── 6. Commercial product names ────────────────────────────────────────────

_COMMERCIAL_HEADS = {'netflix', 'youtube'}
_COMMERCIAL_TAILS = {'gmail', 'google maps', 'instagram', 'youtube'}
_COMMERCIAL_EXACT = {
    ('web', 'enables', 'internet commerce'),  # application domain, not a networking concept
    ('web', 'enables', 'search'),             # same
    ('web', 'enables', 'social network'),     # same
}

def is_commercial(h: str, r: str, t: str) -> bool:
    return (
        h in _COMMERCIAL_HEADS
        or t in _COMMERCIAL_TAILS
        or (h, r, t) in _COMMERCIAL_EXACT
    )


# ─── 7. Relations dropped: < 30 seed triples across all chapters ─────────────
# Threshold set by advisor: keep all relations with ≥ 30 triples ("causes" is
# the cutoff). Dropped: manages, controls, maps_to, decreases, prevents,
# measures, mitigates, routes_to.

_DROPPED_RELATIONS = {
    'manages', 'controls', 'maps_to', 'decreases',
    'prevents', 'measures', 'mitigates', 'routes_to',
}


# ─── 8. Explicit near-duplicates (always remove) ────────────────────────────

_EXPLICIT_DUPES = {
    ('dns',              'provides',   'dns service'),        # tautological
    ('dns',              'provides',   'mapping'),            # too vague; hostname-to-ip is the canonical form
    ('p2p architecture', 'provides',   'selfscalability'),    # portmanteau; self-scaling is the correct form
    ('tcp',              'implements', 'slow-start'),         # hyphen variant; 'slow start' is canonical
    ('quic',             'implements', 'reliable data transfer service'),  # covered by 'reliable data transfer'
}


# ─── 8. Conditional near-duplicate removal ──────────────────────────────────
# For a given (head, relation) pair, if the weaker tail is present AND the
# stronger canonical tail is also present in the same chapter, drop the weaker.

_CONDITIONAL = {
    ('tcp', 'provides'): {
        'reliable data transfer service': 'reliable data transfer',
        'congestion-control mechanism':   'congestion control',
        'connection-oriented service':    'connection-oriented service',  # normalised later
    },
    ('tcp', 'implements'): {
        'congestion control algorithm':     'congestion control',
        'tcp congestion-control algorithm': 'congestion control',
        'end-to-end congestion control':    'congestion control',
        'reliable data transfer service':   'reliable data transfer',
    },
    ('router', 'forwards'): {
        'network-layer datagram': 'ip datagram',
        'datagram':               'ip datagram',
        # 'packet' is intentionally not removed; 'packet' and 'ip datagram' are
        # distinct enough to be kept separately in most chapters
    },
}

def apply_conditional_dedup(triples: list[tuple]) -> list[tuple]:
    existing = {(h, r, t) for h, r, t in triples}
    out = []
    for h, r, t in triples:
        drop_map = _CONDITIONAL.get((h, r), {})
        if t in drop_map:
            canonical = drop_map[t]
            if (h, r, canonical) in existing:
                continue  # weaker variant superseded by canonical
        out.append((h, r, t))
    return out


# ─── Main per-chapter processing ────────────────────────────────────────────

def filter_chapter(chapter: int) -> dict:
    in_csv  = IN_DIR  / f"validated_ch{chapter}.csv"
    out_csv = OUT_DIR / f"validated_ch{chapter}.csv"

    with open(in_csv, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    removed = defaultdict(int)
    triples_out: list[tuple] = []
    seen: set[tuple] = set()

    for row in rows:
        h = rename_entity(normalize(row['head']))
        r = row['relation_type'].strip()
        t = rename_entity(normalize(row['tail']))
        key = (h, r, t)

        if key in seen:
            removed['exact_duplicate'] += 1
            continue
        seen.add(key)

        if is_trivial_isa(h, r, t):
            removed['trivial_isa'] += 1
            continue

        if is_instance(h) or is_instance(t):
            removed['instance_specific'] += 1
            continue

        if key in _WRONG:
            removed['factually_wrong'] += 1
            continue

        if is_alice_bob(h, r, t):
            removed['alice_bob'] += 1
            continue

        if is_commercial(h, r, t):
            removed['commercial'] += 1
            continue

        if key in _EXPLICIT_DUPES:
            removed['near_dupe'] += 1
            continue

        if r in _DROPPED_RELATIONS:
            removed['dropped_relation'] += 1
            continue

        triples_out.append(key)

    before = len(triples_out)
    triples_out = apply_conditional_dedup(triples_out)
    removed['near_dupe'] += before - len(triples_out)

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['head', 'relation_type', 'tail'])
        for h, r, t in triples_out:
            writer.writerow([h, r, t])

    return {
        'chapter':  chapter,
        'in':       len(rows),
        'out':      len(triples_out),
        'removed':  dict(removed),
    }


if __name__ == '__main__':
    total_in = total_out = 0
    grand_removed: dict[str, int] = defaultdict(int)

    for ch in range(1, 9):
        result = filter_chapter(ch)
        n_in, n_out = result['in'], result['out']
        total_in  += n_in
        total_out += n_out
        for reason, count in result['removed'].items():
            grand_removed[reason] += count

        print(f"Ch{ch}: {n_in:4d} → {n_out:4d}  (-{n_in - n_out})")
        for reason, count in sorted(result['removed'].items()):
            if count:
                print(f"       {reason:<24s} -{count}")

    print()
    print(f"Total: {total_in} → {total_out}  (-{total_in - total_out})")
    print()
    for reason, count in sorted(grand_removed.items()):
        print(f"  {reason:<26s} -{count}")
    print(f"\nOutput: {OUT_DIR}")
