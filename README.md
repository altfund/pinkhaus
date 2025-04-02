- local AI w/ webui - https://martech.org/how-to-run-deepseek-locally-on-your-computer/
- start an ollama instance
    - `ollama run deepseek-r1:8b`
    - (it's running at 11434 afterward)
- open webui:
    docker run -d --name open-webui -p 3000:3000 -v open-webui-data:/app/data --pull=always ghcr.io/open-webui/open-webui:main
    - with RAG:
        https://docs.openwebui.com/tutorials/tips/rag-tutorial/

RAG
- letta (RAG)
    docker run   -v ./pgdata:/var/lib/postgresql/data   -p 8283:8283   --env-file .env   letta/letta:latest

- weaviate (RAG)
    https://weaviate.io/blog/local-rag-with-ollama-and-weaviate

