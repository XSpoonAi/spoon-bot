from PIL import Image, ImageDraw

W, H = 256, 256
img = Image.new('RGB', (W, H))
draw = ImageDraw.Draw(img)

# Horizon line
horizon = int(H * 0.58)

# Sky gradient (top -> horizon): deep purple to warm orange
sky_top = (60, 30, 90)
sky_horizon = (255, 140, 70)
for y in range(horizon):
    t = y / max(1, horizon - 1)
    r = int(sky_top[0] * (1 - t) + sky_horizon[0] * t)
    g = int(sky_top[1] * (1 - t) + sky_horizon[1] * t)
    b = int(sky_top[2] * (1 - t) + sky_horizon[2] * t)
    draw.line([(0, y), (W, y)], fill=(r, g, b))

# Ocean gradient (horizon -> bottom): bright reflection blue to deep navy
ocean_top = (40, 120, 170)
ocean_bottom = (10, 30, 70)
for y in range(horizon, H):
    t = (y - horizon) / max(1, (H - horizon - 1))
    r = int(ocean_top[0] * (1 - t) + ocean_bottom[0] * t)
    g = int(ocean_top[1] * (1 - t) + ocean_bottom[1] * t)
    b = int(ocean_top[2] * (1 - t) + ocean_bottom[2] * t)
    draw.line([(0, y), (W, y)], fill=(r, g, b))

# Sun
sun_center = (W // 2, int(H * 0.46))
sun_radius = 28
for r in range(sun_radius, 0, -1):
    t = r / sun_radius
    # soft radial falloff from bright yellow center to orange edge
    color = (
        int(255),
        int(220 - 60 * (1 - t)),
        int(100 - 40 * (1 - t))
    )
    draw.ellipse(
        [sun_center[0] - r, sun_center[1] - r, sun_center[0] + r, sun_center[1] + r],
        fill=color
    )

# Sun reflection on water
for i in range(26):
    y = horizon + i * 3
    half_width = max(2, 22 - i)
    alpha_color = (255, 190, 100)
    draw.line([(sun_center[0] - half_width, y), (sun_center[0] + half_width, y)], fill=alpha_color)

# Gentle wave highlights
for y in range(horizon + 8, H, 8):
    x_offset = (y * 7) % 20
    for x in range(-20 + x_offset, W, 24):
        draw.arc([x, y, x + 16, y + 4], 0, 180, fill=(170, 210, 240))

out_path = r"C:\Users\Ricky\Documents\Project\XSpoonAi\spoon-bot\test_workspace\sunset_ocean_256.png"
img.save(out_path, format='PNG')
print(out_path)
