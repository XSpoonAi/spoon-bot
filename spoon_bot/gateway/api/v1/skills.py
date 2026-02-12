"""Skill management endpoints."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from spoon_bot.gateway.app import get_agent
from spoon_bot.gateway.auth.dependencies import CurrentUser
from spoon_bot.gateway.models.requests import SkillActivateRequest
from spoon_bot.gateway.models.responses import (
    APIResponse,
    MetaInfo,
    SkillListResponse,
    SkillInfo,
    SkillResponse,
)

router = APIRouter()


def _get_skill_manager(agent):
    """Resolve the real SkillManager from the agent.

    The agent's ``.skills`` property returns a plain ``list[str]``,
    so we need the underlying ``_skill_manager`` / ``skill_manager``
    to perform activate/deactivate operations.
    """
    if hasattr(agent, "_skill_manager") and agent._skill_manager is not None:
        return agent._skill_manager
    if hasattr(agent, "skill_manager"):
        sm = agent.skill_manager
        if sm is not None:
            return sm
    return None


def _safe_get_agent():
    """Get agent with a structured error on failure instead of raw 500."""
    try:
        return get_agent()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "AGENT_NOT_READY", "message": "Agent is not initialized yet"},
        )


@router.get("", response_model=APIResponse[SkillListResponse])
async def list_skills(user: CurrentUser) -> APIResponse[SkillListResponse]:
    """List all available skills."""
    request_id = f"req_{uuid4().hex[:12]}"
    agent = _safe_get_agent()

    sm = _get_skill_manager(agent)
    skills: list[SkillInfo] = []

    if sm is not None:
        # Use the real SkillManager registry
        skill_names = sm.list() if hasattr(sm, "list") else []
        for name in skill_names:
            skill = sm.get(name) if hasattr(sm, "get") else None
            if skill:
                is_active = sm.is_active(name) if hasattr(sm, "is_active") else False
                skills.append(
                    SkillInfo(
                        name=skill.name if hasattr(skill, "name") else name,
                        description=getattr(skill, "description", ""),
                        active=is_active,
                    )
                )
    else:
        # Fallback: use the name list from agent.skills property
        for name in getattr(agent, "skills", []):
            skills.append(SkillInfo(name=name, description="", active=False))

    return APIResponse(
        success=True,
        data=SkillListResponse(skills=skills),
        meta=MetaInfo(request_id=request_id),
    )


@router.post("/{skill_name}/activate", response_model=APIResponse[SkillResponse])
async def activate_skill(
    skill_name: str,
    request: SkillActivateRequest,
    user: CurrentUser,
) -> APIResponse[SkillResponse]:
    """Activate a skill."""
    request_id = f"req_{uuid4().hex[:12]}"
    agent = _safe_get_agent()
    sm = _get_skill_manager(agent)

    if sm is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"code": "SKILLS_UNSUPPORTED", "message": "Skill system is not enabled"},
        )

    # Check skill exists
    existing = sm.get(skill_name) if hasattr(sm, "get") else None
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SKILL_NOT_FOUND", "message": f"Skill '{skill_name}' not found"},
        )

    try:
        skill = await sm.activate(skill_name, context=request.context)

        return APIResponse(
            success=True,
            data=SkillResponse(
                activated=True,
                skill=SkillInfo(
                    name=skill.name if hasattr(skill, "name") else skill_name,
                    description=getattr(skill, "description", ""),
                    active=True,
                ),
            ),
            meta=MetaInfo(request_id=request_id),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": str(e)},
        )
    except NotImplementedError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"code": "UNSUPPORTED", "message": str(e)},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "ACTIVATION_FAILED", "message": str(e)},
        )


@router.post("/{skill_name}/deactivate", response_model=APIResponse[SkillResponse])
async def deactivate_skill(
    skill_name: str,
    user: CurrentUser,
) -> APIResponse[SkillResponse]:
    """Deactivate a skill."""
    request_id = f"req_{uuid4().hex[:12]}"
    agent = _safe_get_agent()
    sm = _get_skill_manager(agent)

    if sm is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"code": "SKILLS_UNSUPPORTED", "message": "Skill system is not enabled"},
        )

    # Check skill exists
    existing = sm.get(skill_name) if hasattr(sm, "get") else None
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SKILL_NOT_FOUND", "message": f"Skill '{skill_name}' not found"},
        )

    try:
        deactivated = await sm.deactivate(skill_name)
    except NotImplementedError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"code": "UNSUPPORTED", "message": str(e)},
        )

    return APIResponse(
        success=True,
        data=SkillResponse(deactivated=deactivated),
        meta=MetaInfo(request_id=request_id),
    )
