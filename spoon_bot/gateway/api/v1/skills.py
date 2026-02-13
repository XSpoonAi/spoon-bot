"""Skill management endpoints."""

from __future__ import annotations

from uuid import uuid4
from typing import Any, Protocol, cast

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


class SkillRegistry(Protocol):
    def list(self) -> list[str]:
        ...

    def get(self, name: str) -> Any:
        ...

    def is_active(self, name: str) -> bool:
        ...

    async def activate(self, name: str, context: dict | None = None) -> Any:
        ...

    async def deactivate(self, name: str) -> bool:
        ...


@router.get("", response_model=APIResponse[SkillListResponse])
async def list_skills(user: CurrentUser) -> APIResponse[SkillListResponse]:
    """List all available skills."""
    request_id = f"req_{uuid4().hex[:12]}"
    agent: Any = get_agent()

    skills: list[Any] = []
    skills_obj: Any = getattr(agent, "skills", None)
    if skills_obj is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": "Skill registry unavailable"},
        )

    if isinstance(skills_obj, list):
        skill_names = skills_obj
        for name in skill_names:
            skills.append(
                SkillInfo(
                    name=name,
                    description="",
                    active=False,
                )
            )
    else:
        skills_manager = cast(SkillRegistry, skills_obj)
        skill_names = skills_manager.list()
        for name in skill_names:
            skill = skills_manager.get(name)
            if skill:
                skills.append(
                    SkillInfo(
                        name=skill.name,
                        description=skill.description,
                        active=skills_manager.is_active(name),
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
    agent: Any = get_agent()

    try:
        skills_obj: Any = getattr(agent, "skills", None)
        if skills_obj is None:
            raise ValueError("Skill registry unavailable")
        if isinstance(skills_obj, list):
            raise ValueError("Skill activation not supported by current skill registry")
        if not hasattr(skills_obj, "activate"):
            raise ValueError("Skill activation not supported by current skill registry")

        skills_manager = cast(SkillRegistry, skills_obj)
        skill = await skills_manager.activate(skill_name, context=request.context)

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
    agent: Any = get_agent()

    skills_obj: Any = getattr(agent, "skills", None)
    if skills_obj is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": "Skill registry unavailable"},
        )
    if isinstance(skills_obj, list):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"code": "NOT_SUPPORTED", "message": "Skill deactivation not supported"},
        )
    if not hasattr(skills_obj, "deactivate"):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"code": "NOT_SUPPORTED", "message": "Skill deactivation not supported"},
        )

    skills_manager = cast(SkillRegistry, skills_obj)
    deactivated = await skills_manager.deactivate(skill_name)

    return APIResponse(
        success=True,
        data=SkillResponse(deactivated=deactivated),
        meta=MetaInfo(request_id=request_id),
    )
