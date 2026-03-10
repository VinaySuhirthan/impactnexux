import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io
import time
import math
import sys
import json
import os

HF_API_KEY = "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"
MODEL_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN THEMES
# ─────────────────────────────────────────────────────────────────────────────
THEMES = [
    {
        "name":               "Gold Elite",
        "overlay_top":        (0,   0,   0,  210),
        "overlay_bottom":     (10,  5,   0,  240),
        "vignette_strength":  170,
        "accent_color":       (255, 195,  50, 255),
        "headline_color":     (255, 255, 255, 255),
        "tagline_color":      (215, 195, 140, 240),
        "btn_color":          (212, 160,  10),
        "btn_text_color":     (255, 255, 255),
        "brand_color":        (255, 200,  70, 220),
        "dot_color":          (255, 195,  50, 200),
        "layout":             "centered",
        "accent_style":       "line",
        "brand_text":         "PREMIUM FITNESS",
    },
    {
        "name":               "Electric Blue",
        "overlay_top":        (0,   5,  20,  200),
        "overlay_bottom":     (0,  10,  35,  245),
        "vignette_strength":  150,
        "accent_color":       ( 30, 200, 255, 255),
        "headline_color":     (255, 255, 255, 255),
        "tagline_color":      (160, 220, 255, 240),
        "btn_color":          (  0, 160, 230),
        "btn_text_color":     (255, 255, 255),
        "brand_color":        ( 30, 200, 255, 220),
        "dot_color":          ( 30, 200, 255, 200),
        "layout":             "left_aligned",
        "accent_style":       "double_line",
        "brand_text":         "ELITE PERFORMANCE",
    },
    {
        "name":               "Crimson Power",
        "overlay_top":        (15,  0,   0,  205),
        "overlay_bottom":     (5,   0,   0,  245),
        "vignette_strength":  180,
        "accent_color":       (255,  50,  60, 255),
        "headline_color":     (255, 255, 255, 255),
        "tagline_color":      (255, 185, 185, 240),
        "btn_color":          (210,  30,  40),
        "btn_text_color":     (255, 255, 255),
        "brand_color":        (255,  70,  80, 220),
        "dot_color":          (255,  50,  60, 200),
        "layout":             "bottom_heavy",
        "accent_style":       "bracket",
        "brand_text":         "UNSTOPPABLE",
    },
]

def generate_image(prompt):
    payload  = {"inputs": prompt}
    response = requests.post(MODEL_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return None
    return Image.open(io.BytesIO(response.content))

def parse_prompt(big_prompt):
    headlines, taglines, ctas, image_prompts = [], [], [], []
    section = None
    for line in big_prompt.split("\n"):
        line = line.strip()
        if not line: continue
        if "HEADLINES" in line.upper(): section = "headline"; continue
        if "TAGLINES" in line.upper(): section = "tagline"; continue
        if "CTAS" in line.upper(): section = "cta"; continue
        if "IMAGE_PROMPTS" in line.upper(): section = "image"; continue
        
        m = math.nan # placeholder
        if line[0:1].isdigit() and ". " in line:
            text = line.split(". ", 1)[1]
            if section == "headline": headlines.append(text)
            elif section == "tagline": taglines.append(text)
            elif section == "cta": ctas.append(text)
            elif section == "image": image_prompts.append(text)
    return headlines, taglines, ctas, image_prompts

def load_font(size, bold=False):
    candidates = (
        ["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf"]
    )
    # Check common locations on Windows
    win_fonts = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
    for f in candidates:
        p = os.path.join(win_fonts, f)
        try: return ImageFont.truetype(p, size)
        except: continue
    return ImageFont.load_default()

def wrap_text(text, font, max_width, draw):
    words, lines, current = text.split(), [], ""
    for word in words:
        test = (current + " " + word).strip()
        if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
            current = test
        else:
            if current: lines.append(current)
            current = word
    if current: lines.append(current)
    return lines

def draw_text_block(draw, lines, font, anchor_x, start_y, fill, align="center", spacing=1.2, shadow=True):
    y = start_y
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font)
        w, h = bb[2]-bb[0], bb[3]-bb[1]
        x = anchor_x - w//2 if align == "center" else anchor_x
        if shadow:
            draw.text((x+2, y+2), line, font=font, fill=(0,0,0,160))
        draw.text((x, y), line, font=font, fill=fill)
        y += int(h * spacing)
    return y

def block_h(lines, font, spacing=1.2):
    d = ImageDraw.Draw(Image.new("RGBA", (1,1)))
    h_sum = 0
    for l in lines:
        bb = d.textbbox((0,0), l, font=font)
        h_sum += int((bb[3]-bb[1]) * spacing)
    return h_sum

def vertical_gradient(size, top, bottom):
    w, h = size
    g = Image.new("RGBA", size)
    for y in range(h):
        t = y / (h-1)
        c = tuple(int(top[i]+t*(bottom[i]-top[i])) for i in range(4))
        for x in range(w): g.putpixel((x, y), c)
    return g

def radial_vignette(size, strength=180):
    w, h = size
    v = Image.new("RGBA", size, (0,0,0,0))
    cx, cy = w/2, h/2
    md = math.sqrt(cx**2+cy**2)
    px = v.load()
    for y in range(h):
        for x in range(w):
            a = int(strength*(math.sqrt((x-cx)**2+(y-cy)**2)/md)**1.6)
            px[x,y] = (0,0,0,min(a,255))
    return v

def draw_accent(draw, style, cx, y, color, W):
    if style == "line":
        draw.rectangle([cx-110, y, cx+110, y+4], fill=color)
        return 4
    elif style == "double_line":
        draw.rectangle([cx-130, y, cx+130, y+3], fill=color)
        draw.rectangle([cx-80, y+8, cx+80, y+11], fill=color)
        return 11
    elif style == "bracket":
        lx, rx = cx-140, cx+140
        draw.rectangle([lx, y, lx+28, y+4], fill=color)
        draw.rectangle([lx, y, lx+4, y+22], fill=color)
        draw.rectangle([rx-28, y, rx, y+4], fill=color)
        draw.rectangle([rx-4, y, rx, y+22], fill=color)
        return 22
    return 0

def rounded_rect(draw, x0, y0, x1, y1, r, fill):
    draw.rectangle([x0+r, y0, x1-r, y1], fill=fill)
    draw.rectangle([x0, y0+r, x1, y1-r], fill=fill)
    for ex, ey in [(x0,y0),(x1-2*r,y0),(x0,y1-2*r),(x1-2*r,y1-2*r)]:
        draw.ellipse([ex, ey, ex+2*r, ey+2*r], fill=fill)

def draw_cta_button(draw, cx, y, text, font, btn_color, text_color, px=54, py=20, r=14):
    bb = draw.textbbox((0,0), text, font=font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    bw, bh = tw+px*2, th+py*2
    x0, x1, y1 = cx-bw//2, cx+bw//2, y+bh
    rounded_rect(draw, x0, y, x1, y1, r, btn_color)
    tx, ty = cx-tw//2, y+py
    draw.text((tx+1, ty+1), text, font=font, fill=(0,0,0,110))
    draw.text((tx, ty), text, font=font, fill=text_color)
    return bh

def create_poster(image, headline, tagline, cta, index=0, output_dir="static/generated"):
    W, H = 1080, 1080
    theme = THEMES[index % len(THEMES)]
    base = image.resize((W, H), Image.LANCZOS).convert("RGBA")
    
    # Apply layers
    base = Image.alpha_composite(base, vertical_gradient((W,H), theme["overlay_top"], (0,0,0,0)))
    base = Image.alpha_composite(base, vertical_gradient((W,H), (0,0,0,0), theme["overlay_bottom"]))
    base = Image.alpha_composite(base, radial_vignette((W,H), theme["vignette_strength"]))
    
    draw = ImageDraw.Draw(base)
    fhl = load_font(86, bold=True); ftl = load_font(42); fcta = load_font(46, bold=True); fbrand = load_font(28)
    max_w = int(W * 0.82)
    hl_lines = wrap_text(headline.upper(), fhl, max_w, draw)
    tl_lines = wrap_text(tagline, ftl, max_w, draw)

    cx = W//2
    if theme["layout"] == "centered":
        y = int(H*0.4); y += draw_accent(draw, theme["accent_style"], cx, y, theme["accent_color"], W) + 32
        y = draw_text_block(draw, hl_lines, fhl, cx, y, theme["headline_color"], "center", 1.15) + 40
        y = draw_text_block(draw, tl_lines, ftl, cx, y, theme["tagline_color"], "center", 1.3) + 60
        draw_cta_button(draw, cx, y, cta.upper(), fcta, theme["btn_color"], theme["btn_text_color"])
    elif theme["layout"] == "left_aligned":
        mg = int(W*0.1); y = int(H*0.35)
        y = draw_text_block(draw, hl_lines, fhl, mg, y, theme["headline_color"], "left", 1.15) + 30
        y = draw_text_block(draw, tl_lines, ftl, mg, y, theme["tagline_color"], "left", 1.3) + 50
        draw_cta_button(draw, mg + 150, y, cta.upper(), fcta, theme["btn_color"], theme["btn_text_color"])
    elif theme["layout"] == "bottom_heavy":
        y = int(H*0.15)
        y = draw_text_block(draw, hl_lines, fhl, cx, y, theme["headline_color"], "center", 1.15) + 30
        draw_text_block(draw, tl_lines, ftl, cx, y, theme["tagline_color"], "center", 1.3)
        draw_cta_button(draw, cx, H-150, cta.upper(), fcta, theme["btn_color"], theme["btn_text_color"])

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filename = f"ad_{index+1}_{int(time.time())}.png"
    filepath = os.path.join(output_dir, filename)
    base.convert("RGB").save(filepath, quality=95)
    return filepath

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python imagegen.py <prompt_file_or_text>")
        sys.exit(1)
    
    input_text = sys.argv[1]
    if os.path.exists(input_text):
        with open(input_text, "r", encoding="utf-8") as f:
            big_prompt = f.read()
    else:
        big_prompt = input_text

    headlines, taglines, ctas, image_prompts = parse_prompt(big_prompt)
    count = min(len(headlines), len(taglines), len(ctas), len(image_prompts))
    
    results = []
    for i in range(count):
        img = generate_image(image_prompts[i])
        if img:
            path = create_poster(img, headlines[i], taglines[i], ctas[i], index=i)
            results.append(path.replace("\\", "/"))
    
    print(json.dumps({"images": results}))
