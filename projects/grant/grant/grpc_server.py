"""gRPC server implementation for the SignalService."""

import grpc
import re
from concurrent import futures
import logging

from pinkhaus_models.proto.ominari import external_pb2, external_pb2_grpc
from grant.rag import RAGPipeline, RAGConfig
from grant.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


class SignalServicer(external_pb2_grpc.SignalServiceServicer):
    """Implementation of the SignalService gRPC service."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        vector_store_path: str = "../../resources/fc/grant_chroma_db",
    ):
        self.ollama_client = ollama_client
        self.vector_store_path = vector_store_path
        # Default configuration
        self.default_config = RAGConfig(
            embedding_model="nomic-embed-text",
            llm_model="llama3.2",
            top_k=5,
            temperature=0.7,
        )

    def GetProbabilities(self, request, context):
        """Process a batch of signal requests and return probabilities."""
        print(f"[GRPC] Received batch request with {len(request.requests)} items")
        logger.info(f"Received batch request with {len(request.requests)} items")

        probabilities = []

        for signal_request in request.requests:
            try:
                logger.debug(f"Processing signal request {signal_request.source_id}")
                # Extract the query and model from the request
                query = signal_request.query
                logger.debug(f"Query: {query[:50]}...")
                model = signal_request.model or self.default_config.llm_model
                logger.debug(f"Using model: {model}")

                # Create a custom config with the requested model
                config = RAGConfig(
                    embedding_model=self.default_config.embedding_model,
                    llm_model=model,
                    top_k=self.default_config.top_k,
                    temperature=self.default_config.temperature,
                )
                logger.debug("3")

                # Initialize RAG pipeline
                rag = RAGPipeline(
                    ollama_client=self.ollama_client,
                    config=config,
                    vector_store_path=self.vector_store_path,
                    db_path="../../resources/fc/fc.db",
                )

                # Query the RAG system
                logger.info(f"Processing query: {query[:50]}... with model: {model}")
                print(f"[GRPC] Querying RAG for request {signal_request.source_id}...")

                i = 0
                while True:
                    if i > 0:
                        print(
                            f"[GRPC] Retry {i} for request {signal_request.source_id}"
                        )
                    result = rag.query(query)
                    match = re.search(r"(\d+\.?\d*)", result.answer)
                    probability = float(match.group(1)) if match else None
                    if probability is not None and 0 <= probability <= 1.0:
                        probabilities.append(probability)
                        break
                    elif i == 5:
                        logger.error(
                            "Error processing request grant failed 5 times to produce a probability."
                        )
                        probabilities.append(-1.0)
                        break
                    elif i == 0:
                        query = f"{query} \n\n Remember, the answer must be one decimal number from 0 to 1."
                    elif i == 4:
                        query = f"{query} \n\n A semi-educated guess is fine."
                    logger.error(
                        f"Grant retry {i}, was unable to produce an answer with a probability."
                    )
                    i += 1

            except Exception as e:
                logger.error(f"Error processing request: {e}")
                probabilities.append(-1.0)

        print(f"[GRPC] Returning {len(probabilities)} probabilities: {probabilities}")
        return external_pb2.SignalBatchResponse(probabilities=probabilities)


def serve(port: int = 50051, ollama_base_url: str = "http://localhost:11434"):
    """Start the gRPC server."""
    print(f"[GRPC] Starting server on port {port} with Ollama at {ollama_base_url}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Initialize Ollama client
    ollama_client = OllamaClient(base_url=ollama_base_url)

    # Add the service
    servicer = SignalServicer(ollama_client)
    external_pb2_grpc.add_SignalServiceServicer_to_server(servicer, server)

    # Start the server
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    logger.info(f"gRPC server started on port {port}")
    print(f"[GRPC] Server listening on port {port}")
    print("[GRPC] Server ready to accept connections...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)


if __name__ == "__main__":
    # Only configure logging if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    serve()
