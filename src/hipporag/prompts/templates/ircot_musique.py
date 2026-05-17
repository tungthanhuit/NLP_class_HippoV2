_docs_stanton = (
    "Wikipedia Title: Neville A. Stanton\n"
    "Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of "
    "Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and "
    "Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered "
    "peer-reviewed journal papers on applications of the subject."
)

_docs_southampton = (
    "\n\nWikipedia Title: Southampton\n"
    "The University of Southampton, which was founded in 1862 and received its Royal Charter as a university "
    "in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the "
    "world in the Academic Ranking of World Universities 2010."
)

_demo_question = "When was Neville A. Stanton's employer founded?"

_demo_step1_thought = (
    "Neville A. Stanton is a Professor at the University of Southampton. "
    "The passages do not contain information about when the University of Southampton was founded."
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
    f"{_docs_stanton}\n\n"
    f"Question: {_demo_question}\n"
    '{"thought": "' + _demo_step1_thought + '", '
    '"missing_queries": ["When was the University of Southampton founded?"]}\n\n'
    "--- Example (Step 2: complete information) ---\n"
    f"{_docs_stanton}{_docs_southampton}\n\n"
    f"Question: {_demo_question}\n"
    f"Previous reasoning: {_demo_step1_thought}\n"
    '{"thought": "The University of Southampton was founded in 1862.", "answer": "1862"}'
    "\n"
)

prompt_template = [
    {"role": "system", "content": ircot_system},
    {"role": "user", "content": "${prompt_user}"},
]
