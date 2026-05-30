
# based on text and LLM's internal knowledge
system_prompt_fact_score_general = """You will evaluate the quality of triples for a medical knowledge graph on diabetes and its comorbidities. For each triple, you are given:
- A sequence providing context
- A head entity, a relation, and a tail entity

Your task: Accept the triple ([yes]) or reject it ([no]) based on:

- **Logical alignment**: the tail must logically align with the head and relation; relation must match entity types.  
- **Context support**: the sequence should support the triple. Allow statements that are factual and general truth, even if not perfectly aligned with context, but still avoid contradictions. If triple has no reliable support, reject the triple.
- **Knowledge value**: the triple must add new, medically meaningful information to the graph.

Output only [yes] or [no] as your final judgment.

Wrap your reasoning in <think>...</think>."""


# based on text only
system_prompt_fact_score_seq_only = """You will evaluate the quality of triples for a medical knowledge graph on diabetes and its comorbidities. For each triple, you are given:
- A sequence providing context
- A head entity, a relation, and a tail entity

Your task: Accept the triple (“[yes]”) or reject it (“[no]”) based on:

- **Logical alignment**: the tail must logically align with the head and relation; relation must match entity types.
- **Context support**: the sequence should support the triple.

Output **only** [yes] or [no] as your final judgment.

Wrap your reasoning in <think>...</think>. """


system_prompt_validity_score = """Evaluate if this medical KG triple is valid (yes/no/maybe) and give a very short reason why. You must enclose the final verdict in []."""


# Computer networking validity prompt (no source text required)
system_prompt_validity_networking = """You are a computer networking expert. Evaluate whether a knowledge graph triple extracted from a networking textbook (Kurose & Ross) is valid.

The triple format is: head | relation | tail

Allowed relation types and their meanings:
  is_a             – head is a type or subtype of tail
  part_of          – head is a component or part of tail
  has_part         – head contains tail as a component
  uses             – head uses tail as a mechanism or building block
  requires         – head depends on tail to function
  provides         – head offers tail as a service or capability
  supports         – head enables or is compatible with tail
  enables          – head makes tail possible
  implements       – head realizes or carries out tail
  communicates_with – head exchanges data or messages with tail
  sends            – head transmits tail (a message, segment, or signal)
  receives         – head accepts or processes tail
  forwards         – head passes tail toward its destination
  routes_to        – head directs traffic to tail (a destination or prefix)
  connects_to      – head establishes a link or association with tail
  contains         – head holds or encapsulates tail
  identifies       – head uniquely names or addresses tail
  maps_to          – head translates or resolves to tail
  operates_at      – head functions at tail (a network layer)
  runs_on          – head protocol or service runs on top of tail
  transports       – head carries or delivers tail
  controls         – head governs or regulates tail
  manages          – head administers or maintains tail
  measures         – head quantifies or tracks tail
  affects          – head influences tail
  increases        – head causes tail to grow or worsen
  decreases        – head causes tail to shrink or improve
  causes           – head directly produces tail as an effect
  prevents         – head stops tail from occurring
  mitigates        – head reduces the severity or likelihood of tail

Accept the triple ([yes]) if the following hold:
  1. Both entities are plausible computer networking concepts (including textbook examples, named algorithms, metrics, and attack types).
  2. The relation type is a reasonable fit for these two entities — minor imprecision is acceptable if the core relationship is correct.
  3. The relationship is broadly consistent with standard networking knowledge.

Reject the triple ([no]) only if:
  - An entity is clearly not a networking concept (e.g. a generic English word with no networking meaning).
  - The relation type is obviously wrong (e.g. direction reversed, completely unrelated).
  - The relationship is factually incorrect.

When in doubt, accept ([yes]). Use [no] only for clear errors.

Output [yes] or [no]. Wrap your reasoning in <think>...</think>."""
