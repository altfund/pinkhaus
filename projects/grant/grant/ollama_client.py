import httpx
from typing import Optional, Dict, Any, List, AsyncIterator
from pydantic import BaseModel
import json


class OllamaError(Exception):
    """Base exception for Ollama client errors."""

    pass


class OllamaMessage(BaseModel):
    role: str
    content: str


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    options: Optional[Dict[str, Any]] = None
    format: Optional[str] = None
    raw: bool = False
    keep_alive: Optional[str] = None

    # Support direct option parameters
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    def model_dump(self, **kwargs):
        """Override to handle option parameters."""
        data = super().model_dump(**kwargs)

        # Move temperature, top_k, top_p into options if provided
        options = data.get("options", {}) or {}
        for key in ["temperature", "top_k", "top_p"]:
            if key in data and data[key] is not None:
                options[key] = data.pop(key)

        if options:
            data["options"] = options

        # Remove None values
        return {k: v for k, v in data.items() if v is not None}


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaMessage]
    stream: bool = True
    options: Optional[Dict[str, Any]] = None


class OllamaResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaEmbeddingRequest(BaseModel):
    model: str
    prompt: str


class OllamaEmbeddingResponse(BaseModel):
    embedding: List[float]


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=300.0)
        self.async_client = httpx.AsyncClient(timeout=300.0)

    def __del__(self):
        self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.async_client.aclose()

    def list_models(self) -> Dict[str, Any]:
        response = self.client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return response.json()

    async def list_models_async(self) -> Dict[str, Any]:
        response = await self.async_client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return response.json()

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Any:
        request = OllamaGenerateRequest(
            model=model,
            prompt=prompt,
            stream=stream,
            system=system,
            options=options,
            **kwargs,
        )

        try:
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json=request.model_dump(exclude_none=True),
                timeout=None,
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise OllamaError(f"Failed to connect to Ollama at {self.base_url}: {e}")

        if stream:
            return self._stream_response(response)
        else:
            return OllamaResponse(**response.json())

    async def generate_async(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        request = OllamaGenerateRequest(
            model=model, prompt=prompt, stream=stream, options=options
        )

        response = await self.async_client.post(
            f"{self.base_url}/api/generate",
            json=request.model_dump(exclude_none=True),
            timeout=None,
        )
        response.raise_for_status()

        if stream:
            return self._stream_response_async(response)
        else:
            return OllamaResponse(**response.json())

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        ollama_messages = [OllamaMessage(**msg) for msg in messages]
        request = OllamaChatRequest(
            model=model, messages=ollama_messages, stream=stream, options=options
        )

        response = self.client.post(
            f"{self.base_url}/api/chat",
            json=request.model_dump(exclude_none=True),
            timeout=None,
        )
        response.raise_for_status()

        if stream:
            return self._stream_response(response)
        else:
            return response.json()

    async def chat_async(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Any:
        ollama_messages = [OllamaMessage(**msg) for msg in messages]
        request = OllamaChatRequest(
            model=model, messages=ollama_messages, stream=stream, options=options
        )

        response = await self.async_client.post(
            f"{self.base_url}/api/chat",
            json=request.model_dump(exclude_none=True),
            timeout=None,
        )
        response.raise_for_status()

        if stream:
            return self._stream_response_async(response)
        else:
            return response.json()

    def _stream_response(self, response: httpx.Response):
        for line in response.iter_lines():
            if line:
                yield json.loads(line)

    async def _stream_response_async(
        self, response: httpx.Response
    ) -> AsyncIterator[Dict[str, Any]]:
        async for line in response.aiter_lines():
            if line:
                yield json.loads(line)

    def pull_model(self, model_name: str, stream: bool = True) -> Any:
        response = self.client.post(
            f"{self.base_url}/api/pull",
            json={"name": model_name, "stream": stream},
            timeout=None,
        )
        response.raise_for_status()

        if stream:
            return self._stream_response(response)
        else:
            return response.json()

    async def pull_model_async(self, model_name: str, stream: bool = True) -> Any:
        response = await self.async_client.post(
            f"{self.base_url}/api/pull",
            json={"name": model_name, "stream": stream},
            timeout=None,
        )
        response.raise_for_status()

        if stream:
            return self._stream_response_async(response)
        else:
            return response.json()

    def embeddings(self, model: str, prompt: str) -> OllamaEmbeddingResponse:
        """Generate embeddings for a prompt."""
        request = OllamaEmbeddingRequest(model=model, prompt=prompt)

        try:
            response = self.client.post(
                f"{self.base_url}/api/embeddings",
                json=request.model_dump(),
                timeout=None,
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise OllamaError(f"Failed to connect to Ollama at {self.base_url}: {e}")

        return OllamaEmbeddingResponse(**response.json())

    async def embeddings_async(
        self, model: str, prompt: str
    ) -> OllamaEmbeddingResponse:
        """Generate embeddings for a prompt asynchronously."""
        request = OllamaEmbeddingRequest(model=model, prompt=prompt)

        response = await self.async_client.post(
            f"{self.base_url}/api/embeddings", json=request.model_dump(), timeout=None
        )
        response.raise_for_status()

        return OllamaEmbeddingResponse(**response.json())
