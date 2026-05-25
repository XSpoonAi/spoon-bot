from pathlib import Path


def test_runtime_image_installs_node_and_pnpm_for_skill_clis():
    dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile"
    content = dockerfile.read_text(encoding="utf-8")

    assert "spoon-ai-sdk is pinned from the spoon-core Git repo" in content
    assert "https://deb.nodesource.com/setup_22.x" in content
    assert "apt-get install -y --no-install-recommends nodejs" in content
    assert "npm install -g pnpm" in content
    assert "COPY --from=builder --chown=spoonbot:spoonbot /app/spoon_bot" in content
    assert "chown -R spoonbot:spoonbot /data /app" not in content


def test_runtime_image_normalizes_windows_line_endings_for_entrypoint():
    dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile"
    content = dockerfile.read_text(encoding="utf-8")

    assert "sed -i 's/\\r$//' /app/docker-entrypoint.sh" in content


def test_runtime_image_uses_gateway_default_port():
    dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile"
    content = dockerfile.read_text(encoding="utf-8")

    assert "ENV GATEWAY_PORT=16600" in content
    assert "EXPOSE 16600" in content
    assert "ENV GATEWAY_PORT=8080" not in content
