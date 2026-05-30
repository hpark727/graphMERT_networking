
SYSTEM_CONTEXT = """You are a computer-networking domain extractor building a KG of <head, relation, tail>. You possess advanced knowledge of computer networks and protocols.
Given an input sequence, identify entities specifically relevant to computer networking — protocols, network components, architectural concepts, algorithms, and services — that add meaningful technical knowledge to the KG. Output a Python list of up to 6-word entity "heads" following these rules:

  1. Select a precise and technically specific span (e.g., "transmission control protocol", not "protocol"). Avoid generic terms like "network", "system", "device", "method", "approach", or "technique" without a specific technical context.
  2. Keep original spelling, casing, and abbreviations from the sequence.
  3. Choose only entities that add meaningful networking knowledge to the KG. Do not include general computing terms unrelated to networking (e.g., "operating system", "file system") unless they appear in a networking-specific context.
    A few examples of low-value entities you SHOULD NOT include:
    'large number of packets': "large number" is too context-dependent.
    'various protocols': too generic, no specific protocol named.
    'network performance': too broad, not a specific entity.
    'recent studies': a reference framing, not a networking entity.
    'internet users': a population description, not a networking concept.
  4. If it is not clear whether a term adds networking-specific knowledge, look at the context. Include a term only if the text explicitly links it to a specific networking function, component, or protocol.

You will be provided with incorrect output examples beginning with "Output (Incorrect)". Use them to avoid common mistakes.

Wrap your intermediate reasoning steps clearly within <think> ... </think> tags. Be strict and discard any entity about which you are uncertain.
After generating, verify your output.

Steps:
  1. Identify candidate spans.
  2. Filter by technical precision and networking relevance rules.
  3. Confirm the entity's contribution to the networking KG, discard low-value entities.

Input format:
sequence

Output format:
['head1', 'head2', …]
If none, output []
"""


example_user_1 = """Input:
the internet is a computer network that interconnects billions of computing devices throughout the world. end systems are connected together by a network of communication links and packet switches. there are many types of communication links, which are made up of different types of physical media, including coaxial cable, copper wire, optical fiber, and radio spectrum. different links can transmit data at different rates, with the transmission rate of a link measured in bits / second. when one end system has data to send to another end system, the sending end system segments the data and adds header bytes to each segment. the resulting packages of information, known as packets, are then sent through the network to the destination end system.

Output:"""

example_assistant_1 = """
['end systems', 'packet switches', 'communication links', 'coaxial cable', 'optical fiber', 'packets']
"""

example_explanation_1 = """The rationale behind the output:
Candidates: end systems; communication links; packet switches; coaxial cable; copper wire; optical fiber; radio spectrum; transmission rate; packets.
Keep precise networking entities: end systems (fundamental node type), packet switches (core forwarding device), communication links (architectural concept), coaxial cable and optical fiber (specific physical media), packets (fundamental data unit).
Discard: 'copper wire' (less specific than coaxial cable or optical fiber in networking context), 'transmission rate' (a metric, not an entity), 'radio spectrum' (too broad in this context).
"""

# -----

example_user_2 = """Input:
the internet's principal protocols are known collectively as tcp / ip. tcp stands for transmission control protocol and ip stands for internet protocol. the ip protocol specifies the format of the packets that are sent and received among routers and end systems. the internet's principal protocols are collectively known as tcp / ip. a protocol defines the format and the order of messages exchanged between two or more communicating entities, as well as the actions taken on the transmission and / or receipt of a message or other event. the internet makes use of many other protocols as well, including the domain name system ( dns ) and hyper text transfer protocol ( http ).

Output:"""

example_assistant_2 = """
['tcp / ip', 'transmission control protocol', 'internet protocol', 'domain name system', 'http']
"""

example_explanation_2 = """The rationale behind the output:
Candidates: tcp / ip; transmission control protocol; internet protocol; domain name system; http; protocol; routers.
Keep: tcp / ip (the foundational protocol suite), transmission control protocol, internet protocol (specific named protocols), domain name system and http (specific named application-layer protocols).
Discard: 'protocol' alone is too generic; 'routers' in this passage is incidental context without elaboration.
"""

# -----

example_user_3 = """Input:
the physical layer and data link layer are responsible for handling communication over a single link. the network layer is responsible for moving network - layer packets known as datagrams from one host to another. the internet's network layer includes the celebrated ip protocol, which defines the fields in the datagram as well as how the end systems and routers act on these fields. the transport layer is responsible for transporting application - layer messages between application endpoints. tcp provides a connection - oriented service to its applications. this service includes guaranteed delivery of application - layer messages to the destination and flow control. udp protocol provides a connectionless service to its applications.

Output:"""

example_assistant_3 = """
['physical layer', 'data link layer', 'network layer', 'datagrams', 'ip protocol', 'transport layer', 'flow control', 'udp protocol']
"""

example_explanation_3 = """The rationale behind the output:
Candidates: physical layer; data link layer; network layer; datagrams; ip protocol; transport layer; tcp; flow control; udp protocol; connection-oriented service; connectionless service.
Keep: all five named layers (physical, data link, network, transport) are core OSI/TCP-IP stack concepts; datagrams (specific packet type at network layer); ip protocol; flow control (specific TCP mechanism); udp protocol.
Discard: 'connection-oriented service' and 'connectionless service' are descriptors of service type, not specific named entities. 'tcp' alone is already captured by 'transmission control protocol' context — here 'udp protocol' is the new entity.
"""

# -----

# ==negative examples==
example_user_negative_1 = """Input:
the internet is a computer network that interconnects billions of computing devices throughout the world. end systems are connected together by a network of communication links and packet switches. packets are sent through the network to the destination end system, where they are reassembled into the original data. the internet provides two distinct services to its applications : a connection - oriented service and a connectionless service.

Output (Incorrect):"""

example_assistant_negative_1 = """
['internet', 'computing devices', 'network of communication links', 'packets sent through network', 'two distinct services']
"""

example_explanation_negative_1 = """The above output is incorrect for the following reasons:
Too generic: 'internet' alone is not a specific enough entity for the KG — it is the subject of the entire textbook.
Too generic: 'computing devices' is a vague descriptor, not a specific networking component.
Over-specified phrase: 'network of communication links' is a description, not a named entity — 'communication links' is the precise term.
Predicate phrase: 'packets sent through network' includes a verb phrase; only noun-entity spans should be extracted.
Vague: 'two distinct services' has no specific technical meaning without naming them.

The correct output would be: ['end systems', 'packet switches', 'communication links', 'packets']
"""

# -----

# Positive examples
POSITIVE_PROMPT_EXAMPLES = [
  (example_user_1, example_assistant_1, example_explanation_1),
  (example_user_2, example_assistant_2, example_explanation_2),
  (example_user_3, example_assistant_3, example_explanation_3),
]

# Negative examples
NEGATIVE_PROMPT_EXAMPLES = [
  (example_user_negative_1, example_assistant_negative_1, example_explanation_negative_1),
]
