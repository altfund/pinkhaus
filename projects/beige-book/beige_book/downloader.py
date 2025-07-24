"""
Audio file downloader with support for temporary file handling.
"""

import os
import tempfile
import requests
import hashlib
from typing import Tuple
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


class AudioDownloader:
    """Download audio files from URLs"""

    def __init__(self, chunk_size: int = 8192, timeout: int = 30):
        """
        Initialize the downloader.

        Args:
            chunk_size: Size of chunks to download at a time
            timeout: Request timeout in seconds
        """
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "BeigeBook/1.0 (Podcast Transcriber)"}
        )

    def download_to_temp(
        self, url: str, prefix: str = "beige_book_"
    ) -> Tuple[str, str]:
        """
        Download audio file to a temporary location.

        Args:
            url: URL of the audio file
            prefix: Prefix for the temporary file

        Returns:
            Tuple of (temp_file_path, file_hash)
        """
        logger.info(f"Downloading audio from: {url}")

        try:
            # Create a temporary file
            suffix = self._get_file_extension(url)
            fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)

            try:
                # Download the file
                response = self.session.get(url, stream=True, timeout=self.timeout)
                response.raise_for_status()

                # Get content length for progress tracking
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                # Calculate hash while downloading
                sha256_hash = hashlib.sha256()

                with os.fdopen(fd, "wb") as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            sha256_hash.update(chunk)
                            downloaded += len(chunk)

                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                if (
                                    downloaded % (self.chunk_size * 100) == 0
                                ):  # Log every 100 chunks
                                    logger.debug(f"Download progress: {progress:.1f}%")

                file_hash = sha256_hash.hexdigest()
                logger.info(f"Download complete. Hash: {file_hash}")

                return temp_path, file_hash

            except Exception as e:

            except Exception:
                # Clean up temp file on error
                os.unlink(temp_path)
                raise

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading from {url}: {e}")
            raise

    def _get_file_extension(self, url: str) -> str:
        """
        Get file extension from URL or default to .mp3.

        Args:
            url: URL to extract extension from

        Returns:
            File extension including the dot
        """
        # Try to get extension from URL
        path = Path(url.split("?")[0])  # Remove query parameters
        ext = path.suffix.lower()

        # Common audio extensions
        valid_extensions = {'.mp3', '.mp4', '.m4a', '.ogg', '.wav', '.aac'}

        valid_extensions = {".mp3", ".mp4", ".m4a", ".ogg", ".wav", ".aac"}

        if ext in valid_extensions:
            return ext

        # Default to .mp3
        return ".mp3"

    def cleanup_temp_file(self, temp_path: str):
        """
        Clean up a temporary file.

        Args:
            temp_path: Path to the temporary file
        """
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                logger.debug(f"Cleaned up temp file: {temp_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {temp_path}: {e}")

    def download_with_retry(
        self, url: str, max_retries: int = 5, initial_delay: float = 1.0
    ) -> Tuple[str, str]:
        """
        Download with retry logic and exponential backoff.

        Args:
            url: URL to download
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds

        Returns:
            Tuple of (temp_file_path, file_hash)
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return self.download_to_temp(url)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Calculate exponential backoff with jitter
                    delay = initial_delay * (2**attempt)  # 1s, 2s, 4s, 8s, 16s
                    # Add jitter (±20%) to prevent thundering herd
                    import random

                    jitter = delay * 0.2 * (2 * random.random() - 1)
                    actual_delay = delay + jitter

                    logger.warning(f"Download attempt {attempt + 1} failed, retrying in {actual_delay:.1f}s: {e}")

                    logger.warning(
                        f"Download attempt {attempt + 1} failed, retrying in {actual_delay:.1f}s: {e}"
                    )
                    time.sleep(actual_delay)
                else:
                    logger.error(f"All download attempts failed for: {url}")

        raise last_error