_docs_casablanca = (
    "Wikipedia Title: Casablanca (film)\n"
    "Casablanca is a 1942 American romantic drama film directed by Michael Curtiz and based on the "
    "unproduced stage play Everybody Comes to Rick's by Murray Burnett and Joan Alison. The film stars "
    "Humphrey Bogart, Ingrid Bergman, and Paul Henreid. It was produced by Hal B. Wallis for Warner Bros. "
    "and released on November 26, 1942.\n\n"
    "Wikipedia Title: Michael Curtiz\n"
    "Michael Curtiz (born Manó Kertész Kaminer; December 24, 1886 – April 10, 1962) was a Hungarian-American "
    "film director. He is known for directing Casablanca (1942), The Adventures of Robin Hood (1938), and "
    "White Christmas (1954). He was born in Budapest, Hungary, and emigrated to the United States in 1926."
)

_docs_maltese = (
    "\n\nWikipedia Title: The Maltese Falcon (1941 film)\n"
    "The Maltese Falcon is a 1941 American film noir directed by John Huston in his directorial debut. "
    "It is based on Dashiell Hammett's novel of the same name. The film stars Humphrey Bogart, Mary Astor, "
    "Gladys George, Peter Lorre, and Sydney Greenstreet.\n\n"
    "Wikipedia Title: John Huston\n"
    "John Marcellus Huston (August 5, 1906 – August 28, 1987) was an American film director, screenwriter, "
    "and actor. He was born in Nevada, Missouri, United States. He directed 37 feature films during his "
    "career including The Maltese Falcon, The African Queen, and Chinatown."
)

_demo_question = (
    "Which director was born first, the director of Casablanca or the director of The Maltese Falcon?"
)

_demo_step1_thought = (
    "The director of Casablanca is Michael Curtiz, born on December 24, 1886. "
    "The passages do not contain information about The Maltese Falcon or its director."
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
    f"{_docs_casablanca}\n\n"
    f"Question: {_demo_question}\n"
    '{"thought": "' + _demo_step1_thought + '", '
    '"missing_queries": ["Who directed The Maltese Falcon?", "When was the director of The Maltese Falcon born?"]}\n\n'
    "--- Example (Step 2: complete information) ---\n"
    f"{_docs_casablanca}{_docs_maltese}\n\n"
    f"Question: {_demo_question}\n"
    f"Previous reasoning: {_demo_step1_thought}\n"
    '{"thought": "The director of The Maltese Falcon is John Huston, born on August 5, 1906. '
    'Michael Curtiz was born on December 24, 1886, which is before August 5, 1906.", "answer": "Michael Curtiz"}'
    "\n"
)

prompt_template = [
    {"role": "system", "content": ircot_system},
    {"role": "user", "content": "${prompt_user}"},
]
