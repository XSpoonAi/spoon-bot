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
_ACTIVE_SKILLS: set[str] = set()


def _list_skill_names(agent) -> list[str]:
    skills_obj = getattr(agent, "skills", [])
    if hasattr(skills_obj, "list"):
        try:
            return list(skills_obj.list())
        except Exception:
            return []
    if isinstance(skills_obj, list):
        return skills_obj
    return []


def _get_skill_obj(agent, name: str):
    skills_obj = getattr(agent, "skills", None)
    if hasattr(skills_obj, "get"):
        try:
            return skills_obj.get(name)
        except Exception:
            return None
    return None


@router.get("", response_model=APIResponse[SkillListResponse])
async def list_skills(user: CurrentUser) -> APIResponse[SkillListResponse]:
    """List all available skills."""
    request_id = f"req_{uuid4().hex[:12]}"
    agent = get_agent()

    skill_names = _list_skill_names(agent)
    skills = []

    for name in skill_names:
        skill = _get_skill_obj(agent, name)
        if skill:
            desc = getattr(skill, "description", "")
        else:
            desc = ""

        skills.append(
            SkillInfo(
                name=name,
                description=desc,
                active=(name in _ACTIVE_SKILLS),
            )
        )

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
    """Activate a skill. Works for list-style skills and manager-style skills."""
    request_id = f"req_{uuid4().hex[:12]}"
    agent = get_agent()
    names = _list_skill_names(agent)

    if skill_name not in names:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Skill '{skill_name}' not found"},
        )

    skills_obj = getattr(agent, "skills", None)
    if hasattr(skills_obj, "activate"):
        try:
            maybe_skill = skills_obj.activate(skill_name, context=request.context)
            if hasattr(maybe_skill, "__await__"):
                await maybe_skill
        except Exception:
            # fallback to local active set if manager activation fails
            pass

    _ACTIVE_SKILLS.add(skill_name)

    skill = _get_skill_obj(agent, skill_name)
    return APIResponse(
        success=True,
        data=SkillResponse(
            activated=True,
            skill=SkillInfo(
                name=skill_name,
                description=getattr(skill, "description", "") if skill else "",
                active=True,
            ),
        ),
        meta=MetaInfo(request_id=request_id),
    )


@router.post("/{skill_name}/deactivate", response_model=APIResponse[SkillResponse])
async def deactivate_skill(
    skill_name: str,
    user: CurrentUser,
) -> APIResponse[SkillResponse]:
    """Deactivate a skill. Works for list-style skills and manager-style skills."""
    request_id = f"req_{uuid4().hex[:12]}"
    agent = get_agent()

    names = _list_skill_names(agent)
    if skill_name not in names:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Skill '{skill_name}' not found"},
        )

    skills_obj = getattr(agent, "skills", None)
    if hasattr(skills_obj, "deactivate"):
        try:
            maybe = skills_obj.deactivate(skill_name)
            if hasattr(maybe, "__await__"):
                await maybe
        except Exception:
            pass

    _ACTIVE_SKILLS.discard(skill_name)

    return APIResponse(
        success=True,
        data=SkillResponse(deactivated=True),
        meta=MetaInfo(request_id=request_id),
    )
