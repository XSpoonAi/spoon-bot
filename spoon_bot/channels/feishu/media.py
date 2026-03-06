"""Media upload and download helpers for Feishu/Lark channel.

Wraps lark-oapi image/file APIs in a synchronous helper class.
All methods are synchronous (lark-oapi is a sync SDK) and should be
called via ``asyncio.to_thread()`` from async code.
"""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Any

from loguru import logger

from spoon_bot.channels.feishu.constants import FILE_TYPE_STREAM

if TYPE_CHECKING:
    pass

try:
    from lark_oapi.api.im.v1 import (
        CreateFileRequest,
        CreateFileRequestBody,
        CreateImageRequest,
        CreateImageRequestBody,
        GetImageRequest,
        GetMessageResourceRequest,
    )
    MEDIA_AVAILABLE = True
except ImportError:
    MEDIA_AVAILABLE = False


class FeishuMedia:
    """Synchronous media upload/download helper backed by lark-oapi.

    All public methods are blocking and should be called via
    ``asyncio.to_thread()`` from async context.
    """

    def __init__(self, api_client: Any) -> None:
        self._client = api_client

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_image(self, image_key: str) -> bytes:
        """Download an image by its image_key.

        Args:
            image_key: The image key from message content or upload response.

        Returns:
            Raw image bytes.

        Raises:
            RuntimeError: If the API call fails.
        """
        if not MEDIA_AVAILABLE:
            raise ImportError("lark-oapi im.v1 not available")

        request = GetImageRequest.builder().image_key(image_key).build()
        response = self._client.im.v1.image.get(request)

        if not response.success():
            raise RuntimeError(
                f"Feishu download_image failed: code={response.code}, msg={response.msg}"
            )

        file_obj = response.file
        if hasattr(file_obj, "read"):
            return file_obj.read()
        if isinstance(file_obj, (bytes, bytearray)):
            return bytes(file_obj)
        raise RuntimeError(f"Unexpected image response type: {type(file_obj)}")

    def download_file(
        self, message_id: str, file_key: str, file_type: str = "file"
    ) -> bytes:
        """Download a file/audio/video resource from a message.

        Args:
            message_id: The message ID containing the resource.
            file_key: The file key from message content.
            file_type: Resource type — "image", "file", "audio", "video".

        Returns:
            Raw file bytes.

        Raises:
            RuntimeError: If the API call fails.
        """
        if not MEDIA_AVAILABLE:
            raise ImportError("lark-oapi im.v1 not available")

        request = (
            GetMessageResourceRequest.builder()
            .message_id(message_id)
            .file_key(file_key)
            .type(file_type)
            .build()
        )
        response = self._client.im.v1.message_resource.get(request)

        if not response.success():
            raise RuntimeError(
                f"Feishu download_file failed: code={response.code}, msg={response.msg}"
            )

        file_obj = response.file
        if hasattr(file_obj, "read"):
            return file_obj.read()
        if isinstance(file_obj, (bytes, bytearray)):
            return bytes(file_obj)
        raise RuntimeError(f"Unexpected file response type: {type(file_obj)}")

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_image(
        self, image_data: bytes, image_type: str = "message"
    ) -> str:
        """Upload an image and return its image_key.

        Args:
            image_data: Raw image bytes (PNG, JPG, etc.).
            image_type: "message" (default) or "avatar".

        Returns:
            image_key string for use in subsequent send requests.

        Raises:
            RuntimeError: If the API call fails.
        """
        if not MEDIA_AVAILABLE:
            raise ImportError("lark-oapi im.v1 not available")

        request = (
            CreateImageRequest.builder()
            .request_body(
                CreateImageRequestBody.builder()
                .image_type(image_type)
                .image(io.BytesIO(image_data))
                .build()
            )
            .build()
        )
        response = self._client.im.v1.image.create(request)

        if not response.success():
            raise RuntimeError(
                f"Feishu upload_image failed: code={response.code}, msg={response.msg}"
            )

        return response.data.image_key

    def upload_file(
        self,
        file_data: bytes,
        file_name: str,
        file_type: str = FILE_TYPE_STREAM,
        duration: int = 0,
    ) -> str:
        """Upload a file and return its file_key.

        Args:
            file_data: Raw file bytes.
            file_name: Original filename (used by Feishu for display).
            file_type: One of the FILE_TYPE_* constants (default: "stream").
            duration: Duration in milliseconds for audio/video (default 0).

        Returns:
            file_key string for use in subsequent send requests.

        Raises:
            RuntimeError: If the API call fails.
        """
        if not MEDIA_AVAILABLE:
            raise ImportError("lark-oapi im.v1 not available")

        body_builder = (
            CreateFileRequestBody.builder()
            .file_type(file_type)
            .file_name(file_name)
            .file(io.BytesIO(file_data))
        )
        if duration:
            body_builder = body_builder.duration(duration)

        request = (
            CreateFileRequest.builder()
            .request_body(body_builder.build())
            .build()
        )
        response = self._client.im.v1.file.create(request)

        if not response.success():
            raise RuntimeError(
                f"Feishu upload_file failed: code={response.code}, msg={response.msg}"
            )

        return response.data.file_key

    @staticmethod
    def detect_file_type(file_name: str) -> str:
        """Detect Feishu file_type from file extension.

        Args:
            file_name: Filename with extension.

        Returns:
            Feishu file_type string.
        """
        ext = os.path.splitext(file_name)[1].lower().lstrip(".")
        _ext_map = {
            "opus": "opus",
            "mp4": "mp4",
            "pdf": "pdf",
            "doc": "doc",
            "docx": "doc",
            "xls": "xls",
            "xlsx": "xls",
            "ppt": "ppt",
            "pptx": "ppt",
        }
        return _ext_map.get(ext, FILE_TYPE_STREAM)
