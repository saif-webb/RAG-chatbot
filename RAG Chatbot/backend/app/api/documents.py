"""Document upload, listing and deletion."""

from __future__ import annotations

import hashlib
import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status

from app.api.deps import PipelineDep, SettingsDep, StoreDep
from app.rag.loader import guess_mime, is_supported
from app.schemas.document import DeleteResultOut, DocumentListOut, DocumentOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for ingestion",
)
async def upload_document(
    background: BackgroundTasks,
    store: StoreDep,
    pipeline: PipelineDep,
    settings: SettingsDep,
    file: UploadFile = File(...),
) -> DocumentOut:
    """Accept a file and ingest it in the background.

    Returns 202 immediately: parsing and embedding a large PDF can take a
    minute or more, and holding the request open for it would time out behind
    most proxies. Poll `GET /api/documents` for the status.
    """
    filename = (file.filename or "").strip()
    if not filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The upload has no filename.")

    if not is_supported(filename):
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"'{filename}' is not a supported file type. Upload a PDF, DOCX, "
            f"TXT, MD, CSV or JSON file.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"'{filename}' is empty.")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"'{filename}' is {len(data) / 1e6:.1f} MB, over the "
            f"{settings.max_upload_mb} MB limit.",
        )

    # Hashing the content, not the name, means re-uploading the same file under
    # a different name costs nothing and cannot duplicate citations.
    content_hash = hashlib.sha256(data).hexdigest()
    if existing := await store.get_document_by_hash(content_hash):
        logger.info(
            "'%s' matches existing document %s; skipping re-ingest.",
            filename,
            existing.id,
        )
        return DocumentOut.model_validate(existing).model_copy(update={"duplicate": True})

    document = await store.create_document(
        document_id=uuid4(),
        filename=filename,
        content_hash=content_hash,
        mime_type=file.content_type or guess_mime(filename),
        size_bytes=len(data),
    )
    background.add_task(pipeline.ingest, document.id, filename, data)
    return DocumentOut.model_validate(document)


@router.get("", response_model=DocumentListOut, summary="List documents and status")
async def list_documents(store: StoreDep) -> DocumentListOut:
    documents = await store.list_documents()
    return DocumentListOut(documents=[DocumentOut.model_validate(d) for d in documents])


@router.delete(
    "/{document_id}",
    response_model=DeleteResultOut,
    summary="Delete a document and its chunks",
)
async def delete_document(document_id: UUID, store: StoreDep) -> DeleteResultOut:
    """Delete the document. Its chunks go with it via ON DELETE CASCADE."""
    deleted = await store.delete_document(document_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such document.")
    return DeleteResultOut(id=document_id, deleted=True)
