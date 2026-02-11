"""V1 API router - combines all endpoint modules."""

from fastapi import APIRouter

from spoon_bot.gateway.api.v1.auth import router as auth_router
from spoon_bot.gateway.api.v1.agent import router as agent_router
from spoon_bot.gateway.api.v1.sessions import router as sessions_router
from spoon_bot.gateway.api.v1.tools import router as tools_router
from spoon_bot.gateway.api.v1.skills import router as skills_router

router = APIRouter()

# Include all sub-routers
router.include_router(auth_router, prefix="/auth", tags=["auth"])
router.include_router(agent_router, prefix="/agent", tags=["agent"])
router.include_router(sessions_router, prefix="/sessions", tags=["sessions"])
router.include_router(tools_router, prefix="/tools", tags=["tools"])
router.include_router(skills_router, prefix="/skills", tags=["skills"])
