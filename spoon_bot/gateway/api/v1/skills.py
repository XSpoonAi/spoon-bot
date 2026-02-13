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


@router.get("", response_model=APIResponse[SkillListResponse])
async def list_skills(user: CurrentUser) -> APIResponse[SkillListResponse]:
    """List all available skills."""
    request_id = f"req_{uuid4().hex[:12]}"
    agent = get_agent()

    skill_names = agent.skills.list()
    skills = []

    for name in skill_names:
        skill = agent.skills.get(name)
        if skill:
            skills.append(
                SkillInfo(
                    name=skill.name,
                    description=skill.description,
                    active=agent.skills.is_active(name),
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
    """Activate a skill."""
    request_id = f"req_{uuid4().hex[:12]}"
    agent = get_agent()

    try:
        skill = await agent.skills.activate(skill_name, context=request.context)

        return APIResponse(
            success=True,
            data=SkillResponse(
                activated=True,
                skill=SkillInfo(
                    name=skill.name,
                    description=skill.description,
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
    agent = get_agent()

    deactivated = await agent.skills.deactivate(skill_name)

    return APIResponse(
        success=True,
        data=SkillResponse(deactivated=deactivated),
        meta=MetaInfo(request_id=request_id),
    )
