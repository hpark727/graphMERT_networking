# Gen4 Validated-Both Triple Audit

Manual review of all 2278 triples across Ch1–8. Issues are grouped by type with
specific offending triples listed. Recommendations follow each section.

---

## 1. Trivial `is_a` tails — head `is_a` "concept" / "metric" / "service" / "attack" / "protocol"

These add no information beyond "this entity exists." They are semantically empty
for graph injection since the tail carries no useful representation.

**Ch1:**
- `circuit switching, is_a, concept`
- `network core, is_a, concept`
- `packet switching, is_a, concept`
- `end-point authentication, is_a, concept`
- `protocol layering, is_a, concept`
- `packet loss, is_a, metric`
- `throughput, is_a, metric`
- `processing delay, is_a, metric`
- `propagation delay, is_a, metric`
- `queuing delay, is_a, metric`
- `transmission delay, is_a, metric`
- `end-to-end delay, is_a, metric`
- `end-to-end delay, is_a, delay`
- `remote login, is_a, service`
- `email service, is_a, service`
- `ip spoofing, is_a, attack`
- `ddos attack, is_a, attack`
- `man-in-the-middle attack, is_a, attack`
- `sniffing, is_a, attack`
- `ethernet, is_a, protocol`
- `wifi, is_a, protocol`
- `ncp, is_a, protocol`

**Ch2:**
- `rtt, is_a, metric`
- `upload rate, is_a, metric`
- `peer-to-peer, is_a, service`
- `bit torrent, is_a, protocol`
- `bittorrent, is_a, protocol` ← duplicate with different spacing
- `rarest first, is_a, algorithm`

**Ch3:**
- `physical wire, is_a, channel`
- `ip spoofing, is_a, attack`
- `nak, is_a, packet`
- `ack, is_a, packet`

**Ch5:**
- `hot potato routing, is_a, algorithm`
- `route-selection algorithm, is_a, algorithm`
- `routing algorithm, is_a, algorithm`
- `distance vector algorithm, is_a, algorithm`
- `trap, is_a, pdu`

**Ch6:**
- `sdn-like centralized control, is_a, concept`

**Ch7:**
- `millimeter wave frequencies, is_a, concept`
- `mimo-technology, is_a, concept`
- `wireless medium, is_a, shared medium`

**Ch8:**
- `internet checksum, is_a, checksum`
- `eavesdropping, is_a, attack`
- `ip spoofing, is_a, attack`
- `e‑mail, is_a, service`
- `secure e‑mail system, is_a, service`

**Recommendation:** Filter out all triples where tail ∈ {concept, metric, service,
attack, protocol, algorithm, pdu, channel} with relation `is_a`. These offer no
KG value and dilute embedding quality. ~60 triples.

---

## 2. Instance-specific entities (diagram labels)

These are textbook diagram labels (e.g., "host a", "router r2") that cannot
generalise beyond a single figure. They will never match real text in training.

**Ch1:**
- `buffer, part_of, router a` (should be "router")
- `link, connects_to, router a`
- `link, connects_to, router b`
- `router a, connects_to, router b`
- `router a, forwards, packet`
- `router a, has_part, buffer`
- `router a, receives, packet`
- `router b, forwards, packet`
- `host a, communicates_with, host b`
- `host a, connects_to, link`
- `host a, sends, packet`
- `host b, connects_to, link`
- `host b, sends, packet`
- `host e, receives, packet`

**Ch3:**
- `router r1, forwards, packet`
- `router r2, forwards, packet`
- `router r2, has_part, buffer`
- `router r2, receives, a-c traffic`
- `router r2, receives, b-d traffic`
- `host a, communicates_with, host b` (×many in Ch3)
- `host a, communicates_with, host c`
- `host a, communicates_with, router`
- `host a, connects_to, host b`
- `host b, communicates_with, router`
- `host b, has_part, receive buffer`
- All `host a sends/receives X`, `host b sends/receives X`, `host c/d sends/receives X`

**Ch4:**
- `s1, has_part, flow table`
- `s1, has_part, forwarding table`
- `s2, has_part, flow table`
- `s2, receives, ip datagram`
- `prefix 128.119.40.128/26, part_of, address block`
- `subnet 2, requires, prefix 223.1.17/24`
- `cisco 6500 router, has_part, backplane bus`
- `address block, contains, subnet 1/2/3`
- `home network, contains, address space 10.0.0.0/24`

**Ch5:**
- `router 1b, forwards, packet`
- `router 1b, manages, forwarding table`
- `router 1c, connects_to, router 2a`
- `router 2c, connects_to, router 3a`
- `router 2c, receives, bgp message`
- `router 2c, sends, bgp message`
- `router 3a, sends, bgp message`
- `node 0/1/2/3, sends/receives, routing packet`
- `node 1, communicates_with, node 2`
- `node y, routes_to, node x`
- `node y, sends, distance vector`
- `node y, receives, distance vector`
- `node z, routes_to, node x`
- `node z, sends, distance vector`
- `node z, receives, distance vector`
- `next-hop attribute, identifies, router 2a`
- `next-hop attribute, identifies, router 3d`

**Ch6:**
- `r1, is_a, router` / `r2, is_a, router` / `r5, is_a, router` / `r6, is_a, router`
- `r1, routes_to, destination a`
- `r2, routes_to, destination a`
- `r3, routes_to, destination a/d`
- `mpls infrastructure, connects_to, destination a/d/r5`
- `router r1, receives, ethernet frame`
- `switch s1, has_part, forwarding table`

**Ch7:**
- `h1, connects_to, ap2`
- `h1, receives, beacon frame`
- `node a/b/c/d, communicates_with, node X`
- `station h1, connects_to, access point`
- `station h1, sends, rts frame`
- `station h2, connects_to, access point`
- `ap2, sends, beacon frame`
- `ap2, sends, broadcast ethernet frame`

**Ch8:**
- `r1, sends, ipsec datagram`
- `r2, receives, ipsec datagram`
- `router r1, has_part, security association database`
- `router r2, has_part, security association database`
- `router r2, manages, security association`
- `security association database, part_of, router r2`
- `salesperson laptop, connects_to, headquarters gateway router`
- `salesperson laptop, sends, ipv4 datagram`
- `headquarters gateway router, connects_to, branch-office gateway router`
- `headquarters gateway router, manages, security association`
- `headquarters gateway router, sends, ipsec datagram`
- `headquarters gateway router, sends, ipv4 datagram`
- `branch-office gateway router, manages, security association`

**Recommendation:** Filter all triples where head or tail matches the pattern
of an indexed entity (single letter or letter+digit node labels, "router X",
"host X", "switch sN", "node N", named laptops/sites). ~120 triples.

---

## 3. Naming inconsistency — underscores vs. spaces

The same entity appears with both formats, creating duplicate nodes in the graph.

- `network_interface_card` ↔ `network interface card` (Ch1)
- `data_link_layer` ↔ `data link layer` (Ch1)
- `physical_layer` ↔ `physical layer` (Ch1)
- `transport_layer` ↔ `transport layer` (Ch1)
- `application_layer` ↔ `application layer` (Ch1)

**Specific triples:**
- `network_interface_card, supports, data_link_layer`
- `network_interface_card, supports, physical_layer`
- `transport_layer, provides, reliable_delivery_service`
- `transport_layer, requires, unreliable_edge_to_edge_message_delivery_service`
- `http, operates_at, application_layer`
- `smtp, operates_at, application_layer`
- `ethernet, operates_at, data_link_layer`
- `wifi, operates_at, data_link_layer`

The underscore variants also produce odd tails like
`unreliable_edge_to_edge_message_delivery_service` which is not a real concept name.

**Recommendation:** Normalise all entity names to lowercase with spaces.
Replace underscores with spaces at injection time or in the CSVs.

---

## 4. Factually incorrect or misleading triples

- `packet switching, enables, secure voice` — **FALSE.** Packet switching does not
  inherently enable secure voice. Circuit switching was the historical standard for
  voice; VoIP over packet switching requires additional security protocols.

- `traceroute, runs_on, source host` — **Wrong relation.** `runs_on` is for protocol
  stacking (HTTP runs on TCP). Traceroute is a tool that runs on hosts. Should be
  `source host, runs_on, traceroute` or simply removed — traceroute uses UDP/ICMP.

- `router, causes, packet loss` — **Misleading.** The buffer overflow in a router
  causes packet loss, not the router itself. `output buffer, causes, packet loss` is
  the correct form (also present).

- `packet, increases, delay` — **Vague.** Packets don't increase delay; congestion
  (from high packet arrival rates) does. Confuses cause and carrier.

- `n, increases, distribution time` — **Invalid entity.** "n" is a mathematical
  variable (number of peers in P2P analysis), not a networking concept.

- `three-way tcp handshake, mitigates, playback attack` — **Imprecise.** The
  three-way handshake mitigates replay attacks via sequence numbers, not "playback
  attacks" (non-standard term).

- `tcp fin segment, enables, truncation attack` — **Questionable.** A truncation
  attack exploits premature TCP FIN segments; the FIN segment doesn't "enable" the
  attack in a meaningful causal sense.

- `traffic intensity, decreases, queuing delay` — Appears in Ch1 pre-validation
  (filtered) but check: high traffic intensity increases queuing delay, not decreases.

- `hit rate, measures, web cache` — **Wrong direction.** Hit rate is a property
  measured *of* a web cache; `web cache, measures, hit rate` would be more accurate,
  but even then `identifies` is more appropriate.

---

## 5. Alice/Bob and fictional characters

Named example characters from cryptography chapters are not networking concepts
and will never appear as head entities in text chunks.

**Ch2:**
- `alice, communicates_with, bob`
- `alice, communicates_with, neighboring peer`
- `alice, communicates_with, peer`
- `alice, measures, upload rate`
- `bob, communicates_with, alice`
- `malicious host, sends, spam e‑mail`

**Ch8:**
- `alice, communicates_with, bob`
- `alice, receives, public key`
- `alice, sends, rsa-encrypted message`
- `bob, has_part, private key`
- `bob, receives, rsa-encrypted message`

**Recommendation:** Remove all triples with head/tail = alice, bob, or other
named example characters. ~12 triples.

---

## 6. Non-networking tail concepts (application/service names)

Some tails are web services or products, not networking concepts.

- `http, enables, gmail` — Gmail is an application, not a networking concept
- `http, enables, google maps` — same
- `http, enables, instagram` — same
- `http, enables, youtube` — same
- `netflix, provides, adaptive streaming` — "Netflix" is a company, not a concept
- `netflix, provides, cdn` — same
- `netflix, provides, streaming` — same
- `youtube, provides, streaming` — "YouTube" is not a networking concept
- `web, enables, internet commerce` — "internet commerce" is an application domain
- `web, enables, search` — "search" is not a networking concept
- `web, enables, social network` — "social network" is not a networking concept

**Recommendation:** Remove triples where head or tail is a commercial product/
service name (Netflix, YouTube, Gmail, Google Maps, Instagram). ~11 triples.

---

## 7. Near-duplicate triples (same relationship, minor wording variation)

Many triples express the same fact with slightly different tail phrasing.
These add noise without adding new KG edges.

**TCP reliability (Ch2/Ch3):**
- `tcp, provides, reliable delivery`
- `tcp, provides, reliable data transfer`
- `tcp, provides, reliable data transfer service`
- `tcp, provides, reliable connection-oriented service`
- `tcp, implements, reliable data transfer service`
- `tcp, implements, reliable data transfer`

**DNS (Ch2):**
- `dns, provides, dns service`
- `dns, provides, directory service`
- `dns, provides, hostname-to-ip-address translation service`
- `dns, provides, mapping`

**TCP congestion (Ch3):**
- `tcp, implements, congestion control`
- `tcp, implements, congestion control algorithm`
- `tcp, implements, tcp congestion-control algorithm`
- `tcp, implements, end-to-end congestion control`
- `tcp, provides, congestion control`
- `tcp congestion-control algorithm` + `slow start/fast recovery/congestion avoidance`
  repeated across multiple triples

**Router forwarding (multiple chapters):**
- `router, forwards, packet` (Ch1, 2, 4, 5, 6, 7)
- `router, forwards, datagram` (Ch4, 5, 7)
- `router, forwards, ip datagram` (Ch4, 5, 6, 7, 8)
- `router, forwards, network-layer datagram` (Ch4)

**Recommendation:** De-duplicate by normalising tail synonyms at pre-processing
time. Keep one canonical form per (head, relation) pair per chapter.

---

## 8. Weak structural triples (head `has_part` very generic sub-component)

These are correct but so generic they add minimal KG signal.

- `packet, has_part, header fields`
- `packet, has_part, payload field`
- `packet, contains, source address`
- `packet, contains, destination address`
- `packet, contains, segment`
- `segment, contains, header`
- `url, has_part, hostname`
- `url, has_part, path name`
- `mail message, contains, body`
- `mail message, contains, header`

These are so universally true they likely appear in every chunk involving these
entities and don't discriminate between chapters or topics.

---

## 9. Miscellaneous concerns

- `minitel, supports, x.25` — Minitel is a defunct French teletext service; very
  peripheral to Kurose & Ross and unlikely to appear in any text chunk.

- `ncp, is_a, protocol` — Historical ARPANET protocol; barely covered and no
  meaningful relationships. Single isolated node.

- `docsis 2.0, provides, downstream bitrate` and `docsis 3.0, provides, upstream bitrate`
  — Version-specific triples with no tail value (bitrate is a number, not an entity).

- `source host, measures, round-trip delay` — `measures` will now map to `identifies`
  in gen5. This triple is technically about *traceroute* measuring RTT, not the host
  itself. Should be `traceroute, identifies, round-trip delay` or similar.

- `hit rate, measures, web cache` — Inverted: hit rate is a property of a web cache,
  not something that measures it.

- `central ids processor, receives, ids sensor` — Wrong: a processor receives data
  *from* a sensor, not the sensor itself. Should be `ids sensor, sends, alert` or
  `central ids processor, receives, alert`.

---

## Summary of Recommended Removals

| Category | Approx. count |
|---|---|
| Trivial `is_a` (concept/metric/etc.) | ~60 |
| Instance-specific diagram entities | ~120 |
| Naming inconsistency (underscore variants) | ~8 |
| Factually wrong/misleading | ~8 |
| Alice/Bob fictional characters | ~12 |
| Commercial product names | ~11 |
| Near-duplicates (same fact, different wording) | ~40 |
| **Total flagged** | **~260 (~11%)** |

After removal: estimated **~2020 clean triples**, with higher signal density
and no factual errors.
