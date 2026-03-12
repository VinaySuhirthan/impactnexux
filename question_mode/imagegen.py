import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io
import time
import math
import sys
import json
import os
import re as _re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
MODEL_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
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
    print(f"[DEBUG] Calling HF API for prompt: {prompt[:50]}...")
    try:
        response = requests.post(MODEL_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            print(f"[ERROR] HF API returned status {response.status_code}")
            print(f"[ERROR] Response: {response.text}")
            return None
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"[ERROR] Exception during HF API call: {str(e)}")
        return None

def parse_prompt(big_prompt):
    # Handle literal '\n' strings that might come from CLI arguments
    big_prompt = big_prompt.replace("\\n", "\n")
    print(f"\n[DEBUG] Parsing prompt (length: {len(big_prompt)} characters)")
    
    headlines, taglines, ctas, image_prompts = [], [], [], []
    current_section = None
    
    lines = big_prompt.split("\n")
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Check for section markers
        up = line.upper()
        if "HEADLINE" in up: current_section = "headline"
        elif "TAGLINE" in up: current_section = "tagline"
        elif "CTA" in up: current_section = "cta"
        elif "IMAGE" in up: current_section = "image"
        
        # Extract numbered items (e.g., "1. text" or "Headline: 1. text")
        # Regex to find anything like "1. text"
        match = _re.search(r'\d+\.\s+(.*)', line)
        if match:
            text = match.group(1).strip()
            if current_section == "headline": headlines.append(text)
            elif current_section == "tagline": taglines.append(text)
            elif current_section == "cta": ctas.append(text)
            elif current_section == "image": 
                print(f"[DEBUG] Found image prompt: {str(text)[:50]}...")
                image_prompts.append(text)
    
    print(f"[DEBUG] Parser complete: {len(headlines)} headlines, {len(image_prompts)} image prompts found.")
    return headlines, taglines, ctas, image_prompts

def load_font(size, bold=False):
    print(f"[DEBUG] Loading font (size: {size}, bold: {bold})")
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
    w, h = int(size[0]), int(size[1])
    g = Image.new("RGBA", (w, h))
    for y in range(h):
        t = y / (h - 1) if h > 1 else 0
        c = tuple(int(top[i] + t * (bottom[i] - top[i])) for i in range(4))
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
    print(f"\n[DEBUG] Starting generation of Ad {index+1}")
    W, H = 1080, 1080
    theme = THEMES[index % len(THEMES)]
    print(f"[DEBUG] Theme: {theme['name']} | Layout: {theme['layout']}")
    base = image.resize((W, H), Image.LANCZOS).convert("RGBA")
    
    # Apply layers
    print("[DEBUG] Applying gradient and vignette effects...")
    base = Image.alpha_composite(base, vertical_gradient((W,H), theme["overlay_top"], (0,0,0,0)))
    base = Image.alpha_composite(base, vertical_gradient((W,H), (0,0,0,0), theme["overlay_bottom"]))
    base = Image.alpha_composite(base, radial_vignette((W,H), theme["vignette_strength"]))
    
    draw = ImageDraw.Draw(base)
    # Safe padding so text never bleeds to the edges
    pad = int(W * 0.07)          # 7% padding on each side
    max_w = W - 2 * pad         # usable text width

    # Auto-shrink headline font so it wraps to at most 4 lines
    hl_font_size = 68
    while hl_font_size >= 32:
        fhl = load_font(hl_font_size, bold=True)
        hl_lines = wrap_text(headline.upper(), fhl, max_w, draw)
        if len(hl_lines) <= 4:
            break
        hl_font_size -= 4

    # Auto-shrink tagline font so it wraps to at most 3 lines
    tl_font_size = 36
    while tl_font_size >= 20:
        ftl = load_font(tl_font_size)
        tl_lines = wrap_text(tagline, ftl, max_w, draw)
        if len(tl_lines) <= 3:
            break
        tl_font_size -= 2

    # Auto-shrink CTA font so button fits within image width
    cta_font_size = 42
    while cta_font_size >= 22:
        fcta = load_font(cta_font_size, bold=True)
        bb = draw.textbbox((0, 0), cta.upper(), font=fcta)
        if (bb[2] - bb[0]) + 2 * 54 <= max_w:
            break
        cta_font_size -= 2

    fbrand = load_font(28)

    print(f"[DEBUG] Drawing components for {theme['layout']} layout...")
    cx = W // 2

    if theme["layout"] == "centered":
        y = int(H * 0.38)
        y += draw_accent(draw, theme["accent_style"], cx, y, theme["accent_color"], W) + 28
        y = draw_text_block(draw, hl_lines, fhl, cx, y, theme["headline_color"], "center", 1.15) + 32
        y = draw_text_block(draw, tl_lines, ftl, cx, y, theme["tagline_color"], "center", 1.25) + 48
        draw_cta_button(draw, cx, y, cta.upper(), fcta, theme["btn_color"], theme["btn_text_color"])

    elif theme["layout"] == "left_aligned":
        # Left edge of text block is 'pad'; CTA is centred in the same zone
        left_x = pad
        content_cx = pad + max_w // 2   # centre of the content zone
        y = int(H * 0.33)
        y = draw_text_block(draw, hl_lines, fhl, left_x, y, theme["headline_color"], "left", 1.15) + 28
        y = draw_text_block(draw, tl_lines, ftl, left_x, y, theme["tagline_color"], "left", 1.25) + 44
        draw_cta_button(draw, content_cx, y, cta.upper(), fcta, theme["btn_color"], theme["btn_text_color"])

    elif theme["layout"] == "bottom_heavy":
        y = int(H * 0.15)
        y = draw_text_block(draw, hl_lines, fhl, cx, y, theme["headline_color"], "center", 1.15) + 28
        y = draw_text_block(draw, tl_lines, ftl, cx, y, theme["tagline_color"], "center", 1.25) + 44
        # Place CTA just below tagline but cap so it doesn't overflow bottom
        cta_y = min(y, H - 160)
        draw_cta_button(draw, cx, cta_y, cta.upper(), fcta, theme["btn_color"], theme["btn_text_color"])

    if not os.path.exists(output_dir): 
        print(f"[DEBUG] Creating directory: {output_dir}")
        os.makedirs(output_dir)
    filename = f"ad_{index+1}_{int(time.time())}.png"
    filepath = os.path.join(output_dir, filename)
    print(f"[DEBUG] Saving image to: {filepath}")
    base.convert("RGB").save(filepath, quality=95)
    return filepath

if __name__ == "__main__":
    print("\n" + "="*40)
    print("IMAGEGEN.PY - Ad Generation Starting")
    print("="*40)
    if len(sys.argv) < 2:
        print("[ERROR] No input prompt provided.")
        sys.exit(1)
    
    input_text = sys.argv[1]
    if os.path.exists(input_text):
        print(f"[DEBUG] Loading prompt from file: {input_text}")
        with open(input_text, "r", encoding="utf-8") as f:
            big_prompt = f.read()
    else:
        print("[DEBUG] Using prompt from command line argument.")
        big_prompt = input_text

    headlines, taglines, ctas, image_prompts = parse_prompt(big_prompt)
    count = min(len(headlines), len(taglines), len(ctas), len(image_prompts))
    print(f"[DEBUG] Processing {count} ad variants.")
    
    results = []
    for i in range(count):
        print(f"\n[DEBUG] Fetching background image from Hugging Face for Ad {i+1}...")
        img = generate_image(image_prompts[i])
        if img:
            print(f"[SUCCESS] Image received for Ad {i+1}")
            path = create_poster(img, headlines[i], taglines[i], ctas[i], index=i)
            results.append(path.replace("\\", "/"))
        else:
            print(f"[FAILED] Hugging Face API did not return an image for Ad {i+1}")
    
    print("\n" + "="*40)
    print("FINAL JSON OUTPUT (Required by backend):")
    print(json.dumps({"images": results}))
    print("="*40)