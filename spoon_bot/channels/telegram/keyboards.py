"""InlineKeyboard builders for Telegram bot interactive menus."""

from __future__ import annotations

import math
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from spoon_bot.channels.telegram.constants import (
    AVAILABLE_MODELS,
    CALLBACK_PREFIX,
    MODEL_PROVIDERS,
    MODELS_PER_PAGE,
    MODELS_PER_PROVIDER_PAGE,
    REASONING_LEVELS,
    THINK_LEVELS,
)


def build_model_keyboard(
    page: int = 0,
    current_model: str | None = None,
) -> InlineKeyboardMarkup:
    """Build paginated model selection keyboard.

    Args:
        page: Current page number (0-indexed).
        current_model: Currently active model ID (marked with checkmark).

    Returns:
        InlineKeyboardMarkup with model buttons and pagination.
    """
    total = len(AVAILABLE_MODELS)
    total_pages = math.ceil(total / MODELS_PER_PAGE)
    page = max(0, min(page, total_pages - 1))

    start = page * MODELS_PER_PAGE
    end = start + MODELS_PER_PAGE
    page_models = AVAILABLE_MODELS[start:end]

    # Model buttons (2 per row)
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for model in page_models:
        label = model["name"]
        if model["id"] == current_model:
            label = f"✓ {label}"
        row.append(
            InlineKeyboardButton(
                text=label,
                callback_data=f"{CALLBACK_PREFIX['model']}{model['id']}",
            )
        )
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)

    # Pagination row
    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if page > 0:
            nav.append(
                InlineKeyboardButton(
                    text="◀ Prev",
                    callback_data=f"{CALLBACK_PREFIX['model_page']}{page - 1}",
                )
            )
        nav.append(
            InlineKeyboardButton(
                text=f"{page + 1}/{total_pages}",
                callback_data="noop",
            )
        )
        if page < total_pages - 1:
            nav.append(
                InlineKeyboardButton(
                    text="Next ▶",
                    callback_data=f"{CALLBACK_PREFIX['model_page']}{page + 1}",
                )
            )
        rows.append(nav)

    return InlineKeyboardMarkup(rows)


def build_think_keyboard(current_level: str = "off") -> InlineKeyboardMarkup:
    """Build think level selection keyboard.

    Args:
        current_level: Currently active think level.

    Returns:
        InlineKeyboardMarkup with think level buttons.
    """
    buttons: list[InlineKeyboardButton] = []
    for level, label in THINK_LEVELS.items():
        text = f"✓ {label}" if level == current_level else label
        buttons.append(
            InlineKeyboardButton(
                text=text,
                callback_data=f"{CALLBACK_PREFIX['think']}{level}",
            )
        )
    return InlineKeyboardMarkup([buttons])


def build_verbose_keyboard(current_state: bool = False) -> InlineKeyboardMarkup:
    """Build verbose mode toggle keyboard.

    Args:
        current_state: Current verbose state.

    Returns:
        InlineKeyboardMarkup with on/off buttons.
    """
    on_label = "✓ ON" if current_state else "ON"
    off_label = "✓ OFF" if not current_state else "OFF"
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    text=on_label,
                    callback_data=f"{CALLBACK_PREFIX['verbose']}on",
                ),
                InlineKeyboardButton(
                    text=off_label,
                    callback_data=f"{CALLBACK_PREFIX['verbose']}off",
                ),
            ]
        ]
    )


def build_skill_keyboard(
    skills: list[str],
    page: int = 0,
    per_page: int = 6,
) -> InlineKeyboardMarkup:
    """Build paginated skill browsing keyboard.

    Args:
        skills: List of skill names.
        page: Current page number (0-indexed).
        per_page: Skills per page.

    Returns:
        InlineKeyboardMarkup with skill buttons and pagination.
    """
    if not skills:
        return InlineKeyboardMarkup(
            [[InlineKeyboardButton(text="No skills available", callback_data="noop")]]
        )

    total_pages = math.ceil(len(skills) / per_page)
    page = max(0, min(page, total_pages - 1))

    start = page * per_page
    end = start + per_page
    page_skills = skills[start:end]

    # Skill buttons (1 per row)
    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text=name,
                callback_data=f"{CALLBACK_PREFIX['skill']}{name}",
            )
        ]
        for name in page_skills
    ]

    # Pagination row
    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if page > 0:
            nav.append(
                InlineKeyboardButton(
                    text="◀ Prev",
                    callback_data=f"{CALLBACK_PREFIX['skill_page']}{page - 1}",
                )
            )
        nav.append(
            InlineKeyboardButton(
                text=f"{page + 1}/{total_pages}",
                callback_data="noop",
            )
        )
        if page < total_pages - 1:
            nav.append(
                InlineKeyboardButton(
                    text="Next ▶",
                    callback_data=f"{CALLBACK_PREFIX['skill_page']}{page + 1}",
                )
            )
        rows.append(nav)

    return InlineKeyboardMarkup(rows)


def build_confirm_keyboard(action: str) -> InlineKeyboardMarkup:
    """Build Yes/No confirmation keyboard.

    Args:
        action: Action identifier (e.g. "clear").

    Returns:
        InlineKeyboardMarkup with Yes/No buttons.
    """
    prefix = CALLBACK_PREFIX.get(f"confirm_{action}", f"confirm_{action}:")
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(text="Yes", callback_data=f"{prefix}yes"),
                InlineKeyboardButton(text="No", callback_data=f"{prefix}no"),
            ]
        ]
    )


def build_commands_keyboard(
    commands: list[tuple[str, str]],
    page: int = 0,
    per_page: int = 10,
) -> InlineKeyboardMarkup:
    """Build paginated commands list keyboard.

    Args:
        commands: List of (command, description) tuples.
        page: Current page (0-indexed).
        per_page: Commands per page.

    Returns:
        InlineKeyboardMarkup with command buttons and pagination.
    """
    total_pages = math.ceil(len(commands) / per_page) if commands else 1
    page = max(0, min(page, total_pages - 1))

    start = page * per_page
    end = start + per_page
    page_cmds = commands[start:end]

    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(text=f"/{cmd} - {desc}", callback_data="noop")]
        for cmd, desc in page_cmds
    ]

    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if page > 0:
            nav.append(
                InlineKeyboardButton(
                    text="◀ Prev",
                    callback_data=f"{CALLBACK_PREFIX['commands_page']}{page - 1}",
                )
            )
        nav.append(
            InlineKeyboardButton(
                text=f"{page + 1}/{total_pages}",
                callback_data="noop",
            )
        )
        if page < total_pages - 1:
            nav.append(
                InlineKeyboardButton(
                    text="Next ▶",
                    callback_data=f"{CALLBACK_PREFIX['commands_page']}{page + 1}",
                )
            )
        rows.append(nav)

    return InlineKeyboardMarkup(rows)


def build_reasoning_keyboard(current: str = "off") -> InlineKeyboardMarkup:
    """Build reasoning mode selection keyboard.

    Args:
        current: Current reasoning level ("off", "on", "stream").

    Returns:
        InlineKeyboardMarkup with reasoning level buttons.
    """
    buttons: list[InlineKeyboardButton] = []
    for level, label in REASONING_LEVELS.items():
        text = f"✓ {label}" if level == current else label
        buttons.append(
            InlineKeyboardButton(
                text=text,
                callback_data=f"{CALLBACK_PREFIX['reasoning']}{level}",
            )
        )
    return InlineKeyboardMarkup([buttons])


def build_provider_keyboard(current_model: str | None = None) -> InlineKeyboardMarkup:
    """Build model provider selection keyboard (first level of /models).

    Shows one button per provider with model count, and marks the provider
    containing the currently active model.

    Args:
        current_model: Currently active model ID.

    Returns:
        InlineKeyboardMarkup with provider buttons.
    """
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []

    for provider, models in MODEL_PROVIDERS.items():
        is_current = any(m["id"] == current_model for m in models)
        label = f"✓ {provider} ({len(models)})" if is_current else f"{provider} ({len(models)})"
        row.append(
            InlineKeyboardButton(
                text=label,
                callback_data=f"{CALLBACK_PREFIX['model_provider']}{provider}",
            )
        )
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)

    return InlineKeyboardMarkup(rows)


def build_provider_models_keyboard(
    provider: str,
    page: int = 0,
    current_model: str | None = None,
) -> InlineKeyboardMarkup:
    """Build model list keyboard for a specific provider (second level of /models).

    Args:
        provider: Provider name (key in MODEL_PROVIDERS).
        page: Current page (0-indexed).
        current_model: Currently active model ID.

    Returns:
        InlineKeyboardMarkup with model buttons, pagination, and a Back button.
    """
    models = MODEL_PROVIDERS.get(provider, [])
    total_pages = math.ceil(len(models) / MODELS_PER_PROVIDER_PAGE) if models else 1
    page = max(0, min(page, total_pages - 1))

    start = page * MODELS_PER_PROVIDER_PAGE
    end = start + MODELS_PER_PROVIDER_PAGE
    page_models = models[start:end]

    rows: list[list[InlineKeyboardButton]] = []
    for model in page_models:
        label = model["name"]
        if model["id"] == current_model:
            label = f"✓ {label}"
        rows.append(
            [
                InlineKeyboardButton(
                    text=label,
                    callback_data=f"{CALLBACK_PREFIX['model']}{model['id']}",
                )
            ]
        )

    if total_pages > 1:
        nav: list[InlineKeyboardButton] = []
        if page > 0:
            nav.append(
                InlineKeyboardButton(
                    text="◀ Prev",
                    callback_data=f"{CALLBACK_PREFIX['model_list']}{provider}:{page - 1}",
                )
            )
        nav.append(
            InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop")
        )
        if page < total_pages - 1:
            nav.append(
                InlineKeyboardButton(
                    text="Next ▶",
                    callback_data=f"{CALLBACK_PREFIX['model_list']}{provider}:{page + 1}",
                )
            )
        rows.append(nav)

    # Back button
    rows.append(
        [InlineKeyboardButton(text="<< Back", callback_data=f"{CALLBACK_PREFIX['model_provider']}__back__")]
    )

    return InlineKeyboardMarkup(rows)
