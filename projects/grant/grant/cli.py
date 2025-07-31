import argparse
import sys
from pathlib import Path

from .ollama_client import OllamaClient
from .rag import RAGPipeline, RAGConfig
from .embeddings import EmbeddingService


def list_models(client: OllamaClient) -> None:
    try:
        models = client.list_models()
        if models.get("models"):
            print("Available models:")
            for model in models["models"]:
                print(f"  - {model['name']}")
        else:
            print("No models found. Pull a model using: grant pull <model_name>")
    except Exception as e:
        print(f"Error listing models: {e}", file=sys.stderr)
        sys.exit(1)


def pull_model(client: OllamaClient, model_name: str) -> None:
    try:
        print(f"Pulling model: {model_name}")
        for chunk in client.pull_model(model_name):
            if "status" in chunk:
                print(f"\r{chunk['status']}", end="", flush=True)
        print("\nModel pulled successfully!")
    except Exception as e:
        print(f"\nError pulling model: {e}", file=sys.stderr)
        sys.exit(1)


def query_model(
    client: OllamaClient, model: str, prompt: str, chat: bool = False
) -> None:
    try:
        if chat:
            messages = [{"role": "user", "content": prompt}]
            response = client.chat(model, messages, stream=True)

            print(f"Model: {model}")
            print("Response:")
            for chunk in response:
                if "message" in chunk and "content" in chunk["message"]:
                    print(chunk["message"]["content"], end="", flush=True)
            print()
        else:
            response = client.generate(model, prompt, stream=True)

            print(f"Model: {model}")
            print("Response:")
            for chunk in response:
                if "response" in chunk:
                    print(chunk["response"], end="", flush=True)
            print()
    except Exception as e:
        print(f"Error querying model: {e}", file=sys.stderr)
        sys.exit(1)


def index_command(args):
    """Handle the index command."""
    client = OllamaClient(base_url=args.base_url)

    # Check if database exists
    if not Path(args.db).exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    # Initialize RAG pipeline
    config = RAGConfig(
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    rag = RAGPipeline(
        db_path=args.db,
        ollama_client=client,
        config=config,
        vector_store_path=args.vector_store,
    )

    # Check if embedding model is available
    embeddings = EmbeddingService(client, args.embedding_model)
    if not embeddings.ensure_model_available():
        print(f"Embedding model '{args.embedding_model}' not found.")
        response = input("Would you like to pull it? (y/n): ")
        if response.lower() == "y":
            embeddings.pull_model()
        else:
            sys.exit(1)

    # Index transcriptions
    if args.transcription_id:
        print(f"Indexing transcription {args.transcription_id}...")
        rag.index_transcription(args.transcription_id)
    else:
        rag.index_all_transcriptions(batch_size=args.batch_size)

    # Show stats
    stats = rag.get_stats()
    print("\nIndexing stats:")
    print(f"  Total transcriptions: {stats['total_transcriptions']}")
    print(f"  Indexed transcriptions: {stats['indexed_transcriptions']}")
    print(f"  Total chunks: {stats['total_chunks']}")


def ask_command(args):
    """Handle the ask command."""
    client = OllamaClient(base_url=args.base_url)

    # Check if database exists
    if not Path(args.db).exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    # Initialize RAG pipeline
    config = RAGConfig(
        embedding_model=args.embedding_model,
        llm_model=args.model,
        top_k=args.top_k,
        temperature=args.temperature,
    )

    rag = RAGPipeline(
        db_path=args.db,
        ollama_client=client,
        config=config,
        vector_store_path=args.vector_store,
    )

    # Check if models are available
    embeddings = EmbeddingService(client, args.embedding_model)
    if not embeddings.ensure_model_available():
        print(
            f"Error: Embedding model '{args.embedding_model}' not found.",
            file=sys.stderr,
        )
        print(f"Run: grant pull {args.embedding_model}", file=sys.stderr)
        sys.exit(1)

    # Query
    print(f"Querying: {args.question}")
    print("Searching podcast transcriptions...\n")

    result = rag.query(
        args.question,
        n_results=args.top_k,
        transcription_id=args.transcription_id,
        start_date=args.after,
        end_date=args.before,
    )

    # Display answer
    print("Answer:")
    print("-" * 80)
    print(result.answer)
    print("-" * 80)

    # Display sources if requested
    if args.show_sources:
        print("\nSources:")
        for i, source in enumerate(result.sources, 1):
            print(f"\n{i}. {source['title']}")
            if "timestamp" in source:
                print(f"   Time: {source['timestamp']}")
            if "published" in source:
                print(f"   Published: {source['published']}")
            print(f"   Relevance: {source['score']}")
            print(f"   Preview: {source['text_preview']}")


def stats_command(args):
    """Handle the stats command."""
    client = OllamaClient(base_url=args.base_url)

    # Check if database exists
    if not Path(args.db).exists():
        print(f"Error: Database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    # Initialize RAG pipeline
    rag = RAGPipeline(
        db_path=args.db, ollama_client=client, vector_store_path=args.vector_store
    )

    stats = rag.get_stats()

    print("RAG System Statistics")
    print("=" * 40)
    print(f"Database: {args.db}")
    print(f"Vector store: {args.vector_store}")
    print("\nContent:")
    print(f"  Total transcriptions: {stats['total_transcriptions']}")
    print(f"  Indexed transcriptions: {stats['indexed_transcriptions']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print("\nModels:")
    print(f"  Embedding model: {stats['embedding_model']}")
    print(f"  LLM model: {stats['llm_model']}")
    print(f"  Chunk size: {stats['chunk_size']} tokens")


def main():
    parser = argparse.ArgumentParser(description="Grant - RAG tool using Ollama")
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List models command
    subparsers.add_parser("list", help="List available models")

    # Pull model command
    pull_parser = subparsers.add_parser("pull", help="Pull a model from Ollama")
    pull_parser.add_argument("model", help="Model name to pull")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query a model")
    query_parser.add_argument("model", help="Model to use")
    query_parser.add_argument("prompt", help="Prompt to send to the model")
    query_parser.add_argument(
        "--chat", action="store_true", help="Use chat mode instead of generate mode"
    )

    # Index command
    index_parser = subparsers.add_parser("index", help="Index podcast transcriptions")
    index_parser.add_argument(
        "--db",
        default="../beige-book/protobuf_transcriptions.db",
        help="Path to transcriptions database",
    )
    index_parser.add_argument(
        "--vector-store",
        default="./grant_chroma_db",
        help="Path to vector store directory",
    )
    index_parser.add_argument(
        "--embedding-model", default="nomic-embed-text", help="Embedding model to use"
    )
    index_parser.add_argument(
        "--chunk-size", type=int, default=512, help="Chunk size in tokens"
    )
    index_parser.add_argument(
        "--chunk-overlap", type=int, default=128, help="Chunk overlap in tokens"
    )
    index_parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for indexing"
    )
    index_parser.add_argument(
        "--transcription-id", type=int, help="Index specific transcription ID"
    )

    # Ask command
    ask_parser = subparsers.add_parser(
        "ask", help="Ask questions about podcast content"
    )
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument(
        "--db",
        default="../beige-book/protobuf_transcriptions.db",
        help="Path to transcriptions database",
    )
    ask_parser.add_argument(
        "--vector-store",
        default="./grant_chroma_db",
        help="Path to vector store directory",
    )
    ask_parser.add_argument(
        "--model", default="llama3.2", help="LLM model to use for answering"
    )
    ask_parser.add_argument(
        "--embedding-model", default="nomic-embed-text", help="Embedding model to use"
    )
    ask_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of chunks to retrieve"
    )
    ask_parser.add_argument(
        "--temperature", type=float, default=0.7, help="LLM temperature"
    )
    ask_parser.add_argument(
        "--transcription-id", type=int, help="Limit to specific transcription"
    )
    ask_parser.add_argument("--after", help="Only search after date (YYYY-MM-DD)")
    ask_parser.add_argument("--before", help="Only search before date (YYYY-MM-DD)")
    ask_parser.add_argument(
        "--show-sources", action="store_true", help="Show source information"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show RAG system statistics")
    stats_parser.add_argument(
        "--db",
        default="../beige-book/protobuf_transcriptions.db",
        help="Path to transcriptions database",
    )
    stats_parser.add_argument(
        "--vector-store",
        default="./grant_chroma_db",
        help="Path to vector store directory",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "list":
        client = OllamaClient(base_url=args.base_url)
        list_models(client)
    elif args.command == "pull":
        client = OllamaClient(base_url=args.base_url)
        pull_model(client, args.model)
    elif args.command == "query":
        client = OllamaClient(base_url=args.base_url)
        query_model(client, args.model, args.prompt, args.chat)
    elif args.command == "index":
        index_command(args)
    elif args.command == "ask":
        ask_command(args)
    elif args.command == "stats":
        stats_command(args)


if __name__ == "__main__":
    main()
