from __future__ import annotations

import uuid
from functools import lru_cache

from supabase import Client, create_client

from app.core.config import settings


@lru_cache(maxsize=1)
def _client() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)


def upload_frame(frame_id: str | uuid.UUID, jpeg_bytes: bytes) -> str:
    """Envia o frame JPEG ao bucket "frames" e retorna a URL pública.

    O cv_detector chama este helper diretamente (escreve no Supabase
    sem passar pelo backend), mas o backend também o expõe para uso futuro.
    """
    bucket = settings.SUPABASE_STORAGE_BUCKET
    path = f"{frame_id}.jpg"
    client = _client()

    storage = client.storage.from_(bucket)
    storage.upload(
        path=path,
        file=jpeg_bytes,
        file_options={"content-type": "image/jpeg", "upsert": "true"},
    )
    return storage.get_public_url(path)
