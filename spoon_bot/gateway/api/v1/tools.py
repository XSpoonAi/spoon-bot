"""Tool management endpoints."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from spoon_bot.gateway.app import get_agent
from spoon_bot.gateway.auth.dependencies import CurrentUser, require_scope
from spoon_bot.gateway.models.requests import ToolExecuteRequest
from spoon_bot.gateway.models.responses import (
    APIResponse,
    MetaInfo,
    ToolListResponse,
    ToolInfo,
    ToolResponse,
)

router = APIRouter()


@router.get("", response_model=APIResponse[ToolListResponse])
async def list_tools(user: CurrentUser) -> APIResponse[ToolListResponse]:
    """List all available tools."""
    request_id = f"req_{uuid4().hex[:12]}"
    agent = get_agent()

    tool_names = agent.tools.list_tools()
    tools = []

    for name in tool_names:
        tool = agent.tools.get(name)
        if tool:
            tools.append(
                ToolInfo(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters,
                )
            )

    return APIResponse(
        success=True,
        data=ToolListResponse(tools=tools),
        meta=MetaInfo(request_id=request_id),
    )


@router.get("/{tool_name}/schema")
async def get_tool_schema(
    tool_name: str,
    user: CurrentUser,
) -> dict:
    """Get the schema for a specific tool."""
    agent = get_agent()

    tool = agent.tools.get(tool_name)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Tool '{tool_name}' not found"},
        )

    return {
        "schema": {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
    }


@router.post(
    "/{tool_name}/execute",
    response_model=APIResponse[ToolResponse],
    dependencies=[Depends(require_scope("admin"))],
)
async def execute_tool(
    tool_name: str,
    request: ToolExecuteRequest,
    user: CurrentUser,
) -> APIResponse[ToolResponse]:
    """
    Execute a tool directly.

    Requires admin scope.
    """
    request_id = f"req_{uuid4().hex[:12]}"
    agent = get_agent()

    tool = agent.tools.get(tool_name)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Tool '{tool_name}' not found"},
        )

    try:
        result = await agent.tools.execute(tool_name, **request.arguments)

        return APIResponse(
            success=True,
            data=ToolResponse(result=result, success=True),
            meta=MetaInfo(request_id=request_id),
        )

    except Exception as e:
        return APIResponse(
            success=True,
            data=ToolResponse(result=str(e), success=False),
            meta=MetaInfo(request_id=request_id),
        )
