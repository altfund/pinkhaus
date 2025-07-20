# Pre-Requisites

- Intsall `git subtree`
    - https://github.com/apenwarr/git-subtree/
- Install `flox`
    - https://flox.dev/docs/install-flox/
- Install `docker` (for open-webui and letta)
    - https://docs.docker.com/engine/install/



## AI

### Local AI w/ OpenWebUI and Ollama
- local AI w/ webui - https://martech.org/how-to-run-deepseek-locally-on-your-computer/
- start an ollama instance
    - `ollama run deepseek-r1:8b`
    - (it's running at 11434 afterward)
- open webui:
    docker run -d --name open-webui -p 3000:3000 -v open-webui-data:/app/data --pull=always ghcr.io/open-webui/open-webui:main
    - with RAG:
        https://docs.openwebui.com/tutorials/tips/rag-tutorial/

### RAG
- letta (RAG)
    docker run   -v ./pgdata:/var/lib/postgresql/data   -p 8283:8283   --env-file .env   letta/letta:latest

- weaviate (RAG)
    https://weaviate.io/blog/local-rag-with-ollama-and-weaviate


# on FASTRAG:

- pc: https://news.ycombinator.com/item?id=42174829
yes I believe they're transformers. and yes to the training idea. 
so this argument for fast graph rag made me feel really good. this ground is fairly well trod. putting everything in a database and hoping its useful works but caps out. and so they're pro graphs and relationships. more research revealed there are plenty of approaches we can take, to improve a model's effectiveness with the data it does have. so I think we sprint towards the 0.1 you're imagining then start making it more sophisticated.

- eric:
```python
from fast_graphrag import GraphRAG

DOMAIN = "Evaluate these market odds, match results, and podcast transcripts for soccer and develop a model for what players have an impact on teams performances, which teams beat / underperform their odds, how many goals they score, and what other patterns exist."

EXAMPLE_QUERIES = [
"How many goals are Arsenal likely to score against Nottingham Forest",
"What are the odds that Asenal beat Nottingham Forest?",
"What Arsenal and Nottingham players have an impact on games and how are they likely to react against each other"
]

ENTITY_TYPES = ["Team", "Player", "League", "Manager", "Owner", "Match", "Stadium"]

grag = GraphRAG(
working_dir="./soccer",
domain=DOMAIN,
example_queries="\n".join(EXAMPLE_QUERIES),
entity_types=ENTITY_TYPES
)

with open("./odds_results_and_transcripts.txt") as f:
grag.insert(f.read())

print(grag.query("which team will win [next game]?").response)
```
