"""
Test skill loading and execution via natural language.

Tests:
1. Skill discovery: image_generate and document_export found
2. Trigger matching: natural language triggers correct skill
3. Script execution: skill scripts run and produce output
4. E2E: agent with skills processes natural language requests
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Load .env with override
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

sys.path.insert(0, str(Path(__file__).parent))

# ──────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────
passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


# ──────────────────────────────────────────────────
# Test 1: Skill discovery via SkillManager
# ──────────────────────────────────────────────────


def test_skill_discovery():
    print("\n=== Test 1: Skill Discovery ===\n")

    from spoon_ai.skills.manager import SkillManager

    workspace = Path(__file__).parent / "workspace"
    skill_path = str(workspace / "skills")

    mgr = SkillManager(
        skill_paths=[skill_path],
        auto_discover=True,
        include_default_paths=False,
    )

    all_skills = mgr.list()
    print(f"  Discovered skills: {all_skills}")

    check("At least 2 skills discovered", len(all_skills) >= 2, f"got {len(all_skills)}")
    check("image_generate skill found", "image_generate" in all_skills)
    check("document_export skill found", "document_export" in all_skills)

    # Check image_generate metadata
    img_info = mgr.get_skill_info("image_generate")
    check("image_generate info loaded", img_info is not None)
    if img_info:
        check(
            "image_generate has description",
            bool(img_info.get("description")),
            f"desc={img_info.get('description', '')[:50]}",
        )
        triggers = img_info.get("triggers", [])
        check("image_generate has triggers", len(triggers) > 0, f"got {len(triggers)}")

    # Check document_export metadata
    doc_info = mgr.get_skill_info("document_export")
    check("document_export info loaded", doc_info is not None)
    if doc_info:
        check(
            "document_export has description",
            bool(doc_info.get("description")),
            f"desc={doc_info.get('description', '')[:50]}",
        )

    return mgr


# ──────────────────────────────────────────────────
# Test 2: Keyword/pattern trigger matching
# ──────────────────────────────────────────────────


def test_trigger_matching(mgr):
    print("\n=== Test 2: Trigger Matching ===\n")

    registry = mgr._registry

    # Image generation triggers
    image_triggers = [
        "generate an image of a cat",
        "create image of sunset",
        "draw a picture of a forest",
        "picture of a mountain landscape",
    ]
    for text in image_triggers:
        matches = registry.find_all_matching(text)
        match_names = [m.metadata.name for m in matches]
        check(
            f"'{text[:40]}' triggers image_generate",
            "image_generate" in match_names,
            f"matched: {match_names}",
        )

    # Document export triggers
    doc_triggers = [
        "export this as pdf",
        "generate pdf document",
        "convert to excel spreadsheet",
        "create a markdown document",
    ]
    for text in doc_triggers:
        matches = registry.find_all_matching(text)
        match_names = [m.metadata.name for m in matches]
        check(
            f"'{text[:40]}' triggers document_export",
            "document_export" in match_names,
            f"matched: {match_names}",
        )

    # Non-matching text should NOT trigger these skills
    no_match_texts = [
        "list the files in directory",
        "read the contents of main.py",
    ]
    for text in no_match_texts:
        matches = registry.find_all_matching(text)
        match_names = [m.metadata.name for m in matches]
        img_match = "image_generate" in match_names
        doc_match = "document_export" in match_names
        check(
            f"'{text[:40]}' does NOT trigger img/doc skills",
            not img_match and not doc_match,
            f"matched: {match_names}",
        )


# ──────────────────────────────────────────────────
# Test 3: Skill activation and tool injection
# ──────────────────────────────────────────────────


async def test_skill_activation(mgr):
    print("\n=== Test 3: Skill Activation ===\n")

    # Activate image_generate
    skill = await mgr.activate("image_generate")
    check("image_generate activated", skill is not None)
    check("image_generate is active", mgr.is_active("image_generate"))

    # Check tools from active skills
    tools = mgr.get_active_tools()
    tool_names = [t.name for t in tools]
    print(f"  Active skill tools: {tool_names}")
    check(
        "image_generate has script tool",
        any("image_generate" in n for n in tool_names),
        f"tools: {tool_names}",
    )

    # Get active context (system prompt injection)
    context = mgr.get_active_context()
    check(
        "Active context includes skill instructions",
        "image" in context.lower() if context else False,
        f"context length: {len(context) if context else 0}",
    )

    # Deactivate
    result = await mgr.deactivate("image_generate")
    check("image_generate deactivated", result)
    check("image_generate no longer active", not mgr.is_active("image_generate"))

    # Activate document_export
    skill = await mgr.activate("document_export")
    check("document_export activated", skill is not None)

    tools = mgr.get_active_tools()
    tool_names = [t.name for t in tools]
    check(
        "document_export has script tool",
        any("document_export" in n or "export" in n for n in tool_names),
        f"tools: {tool_names}",
    )

    await mgr.deactivate("document_export")


# ──────────────────────────────────────────────────
# Test 4: Direct script execution (no LLM needed)
# ──────────────────────────────────────────────────


async def test_script_execution():
    print("\n=== Test 4: Direct Script Execution ===\n")

    from spoon_ai.skills.executor import ScriptExecutor

    workspace = Path(__file__).parent / "workspace"
    executor = ScriptExecutor()

    # 4a: Test document_export PDF script
    export_script = workspace / "skills" / "document_export" / "scripts" / "export.py"
    check("export.py script exists", export_script.exists())

    if export_script.exists():
        input_data = json.dumps({
            "content": "Hello World!\nThis is a test PDF generated by spoon-bot skill system.\nLine 3.",
            "format": "pdf",
            "title": "Test PDF",
            "output_path": str(workspace / "exports" / "test_skill.pdf"),
        })

        from spoon_ai.skills.models import SkillScript
        pdf_script = SkillScript(
            name="document_export",
            description="Export to PDF",
            type="python",
            file=str(export_script),
        )

        result = await executor.execute(pdf_script, input_text=input_data, working_directory=str(Path(__file__).parent))
        print(f"  PDF export stdout: {result.stdout[:200]}")

        check("PDF export script executed", result.exit_code == 0, f"exit_code={result.exit_code}, stderr={result.stderr[:100]}")

        if result.stdout:
            try:
                output = json.loads(result.stdout)
                check("PDF export returned success", output.get("success") is True, f"output={output}")
                pdf_path = output.get("file_path", "")
                check("PDF file created", Path(pdf_path).exists() if pdf_path else False, f"path={pdf_path}")
                if Path(pdf_path).exists():
                    size = Path(pdf_path).stat().st_size
                    check(f"PDF has content ({size} bytes)", size > 0)
            except json.JSONDecodeError:
                check("PDF export returned valid JSON", False, f"raw: {result.stdout[:100]}")

    # 4b: Test document_export Markdown script
    input_data = json.dumps({
        "content": "# Test Markdown\n\nThis is a test.",
        "format": "markdown",
        "title": "Test MD",
        "output_path": str(workspace / "exports" / "test_skill.md"),
    })

    md_script = SkillScript(
        name="document_export",
        description="Export to Markdown",
        type="python",
        file=str(export_script),
    )

    result = await executor.execute(md_script, input_text=input_data, working_directory=str(Path(__file__).parent))
    check("Markdown export executed", result.exit_code == 0)
    if result.stdout:
        try:
            output = json.loads(result.stdout)
            check("Markdown export success", output.get("success") is True)
            md_path = output.get("file_path", "")
            check("Markdown file created", Path(md_path).exists() if md_path else False)
        except json.JSONDecodeError:
            check("Markdown export valid JSON", False)

    # 4c: Test image_generate script (network call)
    img_script = workspace / "skills" / "image_generate" / "scripts" / "generate.py"
    check("generate.py script exists", img_script.exists())

    if img_script.exists():
        input_data = json.dumps({
            "prompt": "A simple blue circle on white background",
            "width": 128,
            "height": 128,
            "save_path": str(workspace / "generated_images" / "test_skill.png"),
        })

        gen_script = SkillScript(
            name="image_generate",
            description="Generate image",
            type="python",
            file=str(img_script),
        )

        print("  Generating test image (128x128, may take a few seconds)...")
        result = await executor.execute(gen_script, input_text=input_data, timeout=120, working_directory=str(Path(__file__).parent))
        print(f"  Image gen stdout: {result.stdout[:200]}")

        check("Image gen script executed", result.exit_code == 0, f"exit={result.exit_code}, stderr={result.stderr[:100]}")

        if result.stdout:
            try:
                output = json.loads(result.stdout)
                check("Image gen returned success", output.get("success") is True, f"output={output}")
                img_path = output.get("file_path", "")
                check("Image file created", Path(img_path).exists() if img_path else False, f"path={img_path}")
                if Path(img_path).exists():
                    size = Path(img_path).stat().st_size
                    check(f"Image has content ({size} bytes)", size > 100)
            except json.JSONDecodeError:
                check("Image gen returned valid JSON", False, f"raw: {result.stdout[:100]}")


# ──────────────────────────────────────────────────
# Test 5: E2E — Agent processes natural language with skills
# ──────────────────────────────────────────────────


async def test_e2e_skills():
    print("\n=== Test 5: E2E Agent with Skills ===\n")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip("'\"")
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip("'\"")
    model = os.environ.get("SPOON_BOT_DEFAULT_MODEL", "gpt-5.3-codex")
    provider = os.environ.get("SPOON_BOT_DEFAULT_PROVIDER", "openai")

    if not api_key or api_key.startswith("sk-placeholder"):
        print("  [SKIP] No API key configured, skipping E2E test")
        return

    from spoon_bot.agent.loop import create_agent

    workspace = Path(__file__).parent / "test_workspace"
    workspace.mkdir(exist_ok=True)

    print(f"  Provider: {provider}, Model: {model}")

    # 5a: Create agent WITH skills enabled
    # Use the main workspace so skills are discovered from workspace/skills/
    main_workspace = Path(__file__).parent / "workspace"
    agent = await create_agent(
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        workspace=str(main_workspace),
        enable_skills=True,
        max_iterations=15,
        auto_commit=False,
        tool_profile="coding",  # Use coding profile for broader tool access
    )

    check("Agent created with skills enabled", agent._enable_skills)
    check("SkillManager exists", agent._skill_manager is not None)

    if agent._skill_manager:
        skills = agent._skill_manager.list()
        print(f"  Discovered skills: {skills}")
        check("Skills discovered", len(skills) > 0, f"count={len(skills)}")

    # 5b: Test PDF generation via natural language
    print("\n  Sending PDF generation request...")
    t0 = time.time()
    response = await agent.process(
        "Please generate a PDF file with the title 'Test Report' containing the following content:\n"
        "Line 1: Dynamic tool loading test\n"
        "Line 2: Skills are working correctly\n"
        "Line 3: PDF export via document_export skill\n"
        "Save it as test_report.pdf in the current workspace."
    )
    t1 = time.time()
    print(f"  Response ({t1-t0:.1f}s): {response[:200]}...")

    # Check if PDF was created (may be in workspace/exports/ or workspace root)
    pdf_candidates = list(main_workspace.rglob("*.pdf"))
    check(
        "PDF file created via skill",
        len(pdf_candidates) > 0,
        f"found: {[str(p.name) for p in pdf_candidates]}",
    )

    # 5c: Test image generation via natural language
    print("\n  Sending image generation request...")
    t0 = time.time()
    response = await agent.process(
        "Generate an image of a beautiful sunset over the ocean. "
        "Use 256x256 resolution and save it in the workspace."
    )
    t1 = time.time()
    print(f"  Response ({t1-t0:.1f}s): {response[:200]}...")

    # Check if image was created
    img_candidates = list(main_workspace.rglob("*.png"))
    check(
        "Image file created via skill",
        len(img_candidates) > 0,
        f"found: {[str(p.name) for p in img_candidates]}",
    )


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────


def main():
    global passed, failed

    print("=" * 60)
    print("Skill System Test Suite")
    print("=" * 60)

    # Test 1: Discovery
    mgr = test_skill_discovery()

    # Test 2: Trigger matching
    test_trigger_matching(mgr)

    # Test 3: Activation
    asyncio.run(test_skill_activation(mgr))

    # Test 4: Script execution (direct)
    asyncio.run(test_script_execution())

    # Test 5: E2E with LLM
    asyncio.run(test_e2e_skills())

    # Summary
    total = passed + failed
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
