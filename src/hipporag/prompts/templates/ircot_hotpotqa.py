_docs_nobody_loves_you = (
    "Wikipedia Title: Nobody Loves You (When You're Down and Out)\n"
    "\"Nobody Loves You (When You're Down and Out)\" is a song written by John Lennon released on his "
    "1974 album \"Walls and Bridges\". The song is included on the 1986 compilation \"Menlove Ave.\", "
    "the 1990 boxset \"Lennon\", the 1998 boxset \"John Lennon Anthology\", the 2005 two-disc compilation, "
    "and the 2010 boxset \"Gimme Some Truth\"."
)

_docs_walls_and_bridges = (
    "\n\nWikipedia Title: Walls and Bridges\n"
    "Walls and Bridges is the fifth studio album by English musician John Lennon. It was issued by Apple "
    "Records on 26 September 1974 in the United States and on 4 October in the United Kingdom. Written, "
    "recorded and released during his 18-month separation from Yoko Ono, the album captured Lennon in the "
    "midst of his \"Lost Weekend\"."
)

_demo_question = (
    "Nobody Loves You was written by John Lennon and released on what album that was issued by Apple "
    "Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?"
)

_demo_step1_thought = (
    "The song Nobody Loves You was written by John Lennon. "
    "The passages do not specify which album it was released on or its Apple Records release details."
)

ircot_system = (
    "You are an intelligent assistant for multi-hop reasoning over document collections. "
    "Given retrieved passages, a question, and any previous reasoning, generate one reasoning step.\n\n"
    "RESPONSE FORMAT — always respond with valid JSON, exactly one of:\n"
    '  {"thought": "<what you found>", "missing_queries": ["<question about missing entity 1>", "<question about missing entity 2>", ...]}\n'
    '  {"thought": "<final reasoning>", "answer": "<concise final answer>"}\n\n'
    "Rules:\n"
    '  - Use "missing_queries" when the passages lack information needed to answer. '
    "Generate one targeted question per missing entity or missing fact — each must be a complete question sentence (e.g. 'Who directed X?', 'When was Y born?'), NOT a noun phrase or keyword fragment.\n"
    '  - Use "answer" only when the passages contain explicit evidence for the complete answer. '
    "Never guess or use knowledge outside the provided passages.\n"
    '  - "thought" must summarise what you found and (for missing_queries) which entities are still missing.\n\n'
    "--- Example (Step 1: partial information) ---\n"
    f"{_docs_nobody_loves_you}\n\n"
    f"Question: {_demo_question}\n"
    '{"thought": "' + _demo_step1_thought + '", '
    '"missing_queries": ["Which John Lennon album was issued by Apple Records and recorded during his 18-month separation from Yoko Ono?"]}\n\n'
    "--- Example (Step 2: complete information) ---\n"
    f"{_docs_nobody_loves_you}{_docs_walls_and_bridges}\n\n"
    f"Question: {_demo_question}\n"
    f"Previous reasoning: {_demo_step1_thought}\n"
    '{"thought": "Walls and Bridges was issued by Apple Records during John Lennon\'s 18-month separation '
    'from Yoko Ono. Nobody Loves You was released on Walls and Bridges.", "answer": "Walls and Bridges"}'
    "\n"
)

prompt_template = [
    {"role": "system", "content": ircot_system},
    {"role": "user", "content": "${prompt_user}"},
]
