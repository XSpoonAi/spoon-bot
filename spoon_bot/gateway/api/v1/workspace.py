"""Workspace file-tree endpoints."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from spoon_bot.gateway.app import get_agent
from spoon_bot.gateway.auth.dependencies import CurrentUser
from spoon_bot.gateway.models.responses import APIResponse, MetaInfo

router = APIRouter()


class TreeNode(BaseModel):
    """Single node in the directory tree."""

    name: str
    path: str
    type: str  # "file" or "directory"
    size: int | None = None
    children: list["TreeNode"] | None = None


def _build_tree(
    root: Path,
    *,
    max_depth: int = 5,
    current_depth: int = 0,
    include_hidden: bool = False,
) -> list[TreeNode]:
    """Recursively build a directory tree, respecting depth and hidden-file filters."""
    if current_depth >= max_depth or not root.is_dir():
        return []

    nodes: list[TreeNode] = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return []

    for entry in entries:
        if not include_hidden and entry.name.startswith("."):
            continue
        if entry.name in ("__pycache__", "node_modules", ".git"):
            continue

        rel = str(entry.relative_to(root.parent)).replace("\\", "/")

        if entry.is_dir():
            children = _build_tree(
                entry,
                max_depth=max_depth,
                current_depth=current_depth + 1,
                include_hidden=include_hidden,
            )
            nodes.append(TreeNode(
                name=entry.name,
                path=rel,
                type="directory",
                children=children,
            ))
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = None
            nodes.append(TreeNode(
                name=entry.name,
                path=rel,
                type="file",
                size=size,
            ))

    return nodes


@router.get("/tree", response_model=APIResponse[list[TreeNode]])
async def get_workspace_tree(
    user: CurrentUser,
    path: str = Query("", description="Sub-path relative to workspace root"),
    depth: int = Query(3, ge=1, le=10, description="Maximum directory depth"),
    include_hidden: bool = Query(False, description="Include hidden files/dirs"),
) -> APIResponse[list[TreeNode]]:
    """Return the workspace directory tree."""
    request_id = f"req_{uuid4().hex[:12]}"
    agent = get_agent()

    workspace = Path(getattr(agent, "workspace", Path.home() / ".spoon-bot" / "workspace"))
    target = (workspace / path).resolve()

    if not str(target).startswith(str(workspace.resolve())):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "PATH_OUTSIDE_WORKSPACE", "message": "Path is outside workspace"},
        )

    if not target.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "PATH_NOT_FOUND", "message": f"Path not found: {path}"},
        )

    tree = _build_tree(target, max_depth=depth, include_hidden=include_hidden)

    return APIResponse(
        success=True,
        data=tree,
        meta=MetaInfo(request_id=request_id),
    )
