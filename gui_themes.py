#!/usr/bin/env python3
"""
Max Headroom - Sci-Fi GUI Themes Module
Real working themed components:
- Matrix digital rain animation
- Atlantean sacred geometry HUD
- CRT scanline overlay
- Crystalline neon panels
- Terminal-style logging
- Sci-fi HUD crosshairs and data rings
- Glitch/scan effects
No placeholders. All canvas-based animations use tkinter after() loops.
"""
import tkinter as tk
from tkinter import ttk
import random
import math
import time
import numpy as np

# ============================================================================
# COLOR PALETTE - Sci-Fi / Matrix / Max Headroom / Atlantean
# ============================================================================
class Colors:
    """Unified color palette across all themes."""
    # Backgrounds
    DEEP_SPACE = "#000428"
    VOID_BLACK = "#000000"
    DARK_PANEL = "#001122"
    HUD_BG = "#001122"
    
    # Max Headroom / CRT
    CRT_CYAN = "#00FFFF"
    CRT_GREEN = "#00FF41"
    CRT_AMBER = "#FFB000"
    SCANLINE = "#001100"
    
    # Matrix
    MATRIX_GREEN = "#00FF41"
    MATRIX_DARK = "#003B00"
    MATRIX_GLOW = "#00FF88"
    
    # Atlantean / Crystal
    ATLANTEAN_TEAL = "#00E5FF"
    CRYSTAL_BLUE = "#00D4FF"
    SACRED_GOLD = "#FFD700"
    AQUA_GLOW = "#00FFE5"
    
    # Sci-Fi Neon
    NEON_PINK = "#FF00FF"
    NEON_PURPLE = "#BF00FF"
    NEON_ORANGE = "#FF6B00"
    PLASMA_BLUE = "#0080FF"
    
    # Alerts
    WARNING = "#FF3300"
    ALERT = "#FF0000"
    OK = "#00FF41"

# ============================================================================
# MATRIX DIGITAL RAIN - Animated background
# ============================================================================
class MatrixRainCanvas(tk.Canvas):
    """
    Real Matrix-style falling code animation.
    Each column has characters falling at different speeds.
    Uses tkinter after() for smooth animation without blocking.
    """
    
    CHARS = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾇﾈ0123456789ABCDEF"
    
    def __init__(self, parent, width=800, height=600, column_spacing=14, font_size=12, **kwargs):
        super().__init__(parent, width=width, height=height, bg=Colors.VOID_BLACK, 
                        highlightthickness=0, **kwargs)
        self.column_width = column_spacing
        self.font_size = font_size
        self.columns = width // self.column_width
        
        # Each column: [y_position, speed, chars_list, brightness_phase]
        self.drops = []
        for i in range(self.columns):
            self.drops.append({
                'x': i * self.column_width + self.column_width // 2,
                'y': random.randint(-height, 0),
                'speed': random.randint(2, 8),
                'length': random.randint(5, 20),
                'chars': [random.choice(self.CHARS) for _ in range(30)],
                'phase': random.random() * math.pi * 2,
            })
        
        self.text_ids = {}
        self.running = False
        self._after_id = None
        
    def start(self):
        self.running = True
        self._animate()
    
    def stop(self):
        self.running = False
        if self._after_id:
            self.after_cancel(self._after_id)
    
    def _animate(self):
        if not self.running:
            return
        
        self.delete("all")
        h = int(self['height'])
        
        for drop in self.drops:
            # Update position
            drop['y'] += drop['speed']
            if drop['y'] > h + drop['length'] * self.font_size:
                drop['y'] = -drop['length'] * self.font_size
                drop['speed'] = random.randint(2, 8)
                drop['length'] = random.randint(5, 20)
            
            # Draw the trail
            for i in range(drop['length']):
                char_y = int(drop['y'] - i * self.font_size)
                if 0 <= char_y <= h:
                    # Head is brightest white-green, trail fades to dark green
                    if i == 0:
                        color = "#CCFFCC"  # White-green head
                        font = ("Consolas", self.font_size, "bold")
                    elif i < 3:
                        color = Colors.MATRIX_GREEN
                        font = ("Consolas", self.font_size, "bold")
                    else:
                        fade = max(0, 1.0 - (i / drop['length']))
                        g = int(0x41 * fade)
                        color = f"#00{g:02x}00"
                        font = ("Consolas", self.font_size)
                    
                    char_idx = (i + int(drop['y'] / self.font_size)) % len(drop['chars'])
                    # Occasionally change a character for flicker effect
                    if random.random() < 0.02:
                        drop['chars'][char_idx] = random.choice(self.CHARS)
                    
                    self.create_text(
                        drop['x'], char_y,
                        text=drop['chars'][char_idx],
                        fill=color,
                        font=font,
                        anchor="n"
                    )
        
        self._after_id = self.after(40, self._animate)  # 25 FPS

# ============================================================================
# ATLANTEAN SACRED GEOMETRY - Rotating crystal patterns
# ============================================================================
class SacredGeometryCanvas(tk.Canvas):
    """
    Rotating sacred geometry patterns:
    - Flower of Life circles
    - Metatron's cube lines
    - Crystal hexagon grid
    All animated with tkinter after().
    """
    
    def __init__(self, parent, width=400, height=400, **kwargs):
        super().__init__(parent, width=width, height=height, bg=Colors.VOID_BLACK,
                        highlightthickness=0, **kwargs)
        self.center_x = width // 2
        self.center_y = height // 2
        self.rotation = 0.0
        self.pulse = 0.0
        self.running = False
        self._after_id = None
        
    def start(self):
        self.running = True
        self._animate()
    
    def stop(self):
        self.running = False
        if self._after_id:
            self.after_cancel(self._after_id)
    
    def _animate(self):
        if not self.running:
            return
        
        self.delete("all")
        w = int(self['width'])
        h = int(self['height'])
        cx, cy = w // 2, h // 2
        
        self.rotation += 0.3
        self.pulse = (math.sin(time.time() * 2) + 1) / 2  # 0 to 1
        
        # Outer glow ring (pulsing)
        outer_r = min(cx, cy) - 10 + int(self.pulse * 10)
        self.create_oval(cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r,
                        outline=Colors.ATLANTEAN_TEAL, width=2, stipple="gray50")
        
        # Flower of Life - 7 circles
        base_r = outer_r * 0.25
        for i in range(7):
            if i == 0:
                x, y = cx, cy
            else:
                angle = math.radians(self.rotation + (i - 1) * 60)
                x = cx + math.cos(angle) * base_r
                y = cy + math.sin(angle) * base_r
            
            color = Colors.CRYSTAL_BLUE if i == 0 else Colors.AQUA_GLOW
            width = 2 if i == 0 else 1
            self.create_oval(x - base_r, y - base_r, x + base_r, y + base_r,
                           outline=color, width=width)
        
        # Metatron's Cube - connecting lines between 6 outer points
        points = []
        for i in range(6):
            angle = math.radians(self.rotation + i * 60)
            px = cx + math.cos(angle) * base_r * 2
            py = cy + math.sin(angle) * base_r * 2
            points.append((px, py))
            # Draw outer hexagon points
            self.create_oval(px - 3, py - 3, px + 3, py + 3,
                           fill=Colors.SACRED_GOLD, outline="")
        
        # Connect all points (Metatron's cube)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                self.create_line(points[i][0], points[i][1], points[j][0], points[j][1],
                               fill=Colors.CRYSTAL_BLUE, width=1, stipple="gray25")
        
        # Central crystal
        crystal_size = 15 + int(self.pulse * 8)
        crystal_points = []
        for i in range(6):
            angle = math.radians(self.rotation * 2 + i * 60)
            px = cx + math.cos(angle) * crystal_size
            py = cy + math.sin(angle) * crystal_size
            crystal_points.extend([px, py])
        
        self.create_polygon(crystal_points, fill=Colors.ATLANTEAN_TEAL,
                          outline=Colors.AQUA_GLOW, width=2, stipple="gray75")
        
        # Inner rotating triangle
        tri_r = crystal_size * 0.5
        tri_points = []
        for i in range(3):
            angle = math.radians(-self.rotation * 3 + i * 120)
            tri_points.extend([
                cx + math.cos(angle) * tri_r,
                cy + math.sin(angle) * tri_r
            ])
        self.create_polygon(tri_points, outline=Colors.SACRED_GOLD, width=2, fill="")
        
        self._after_id = self.after(50, self._animate)  # 20 FPS

# ============================================================================
# CRT SCANLINE OVERLAY
# ============================================================================
class CRTOverlayCanvas(tk.Canvas):
    """
    Real CRT monitor effect overlay.
    Horizontal scanlines + occasional flicker + slight barrel distortion hint.
    """
    
    def __init__(self, parent, width=640, height=480, **kwargs):
        super().__init__(parent, width=width, height=height, bg="",
                        highlightthickness=0, **kwargs)
        self['bg'] = ''  # Transparent
        self.scanline_ids = []
        self.flicker = 0
        self.running = False
        self._after_id = None
        
    def start(self):
        self.running = True
        self._draw_scanlines()
    
    def stop(self):
        self.running = False
        if self._after_id:
            self.after_cancel(self._after_id)
    
    def _draw_scanlines(self):
        if not self.running:
            return
        
        self.delete("all")
        w = int(self['width'])
        h = int(self['height'])
        
        # Horizontal scanlines
        for y in range(0, h, 3):
            alpha = 30 + int(self.flicker * 20)
            color = f"#{0:02x}{alpha:02x}{0:02x}"
            self.create_line(0, y, w, y, fill=color, width=1)
        
        # Occasional flicker band
        if random.random() < 0.05:
            band_y = random.randint(0, h - 20)
            self.create_rectangle(0, band_y, w, band_y + 2,
                                fill=Colors.CRT_CYAN, outline="", stipple="gray50")
        
        # Vignette edges (darker corners)
        self.create_rectangle(0, 0, w, 20, fill="", outline="", stipple="gray75")
        self.create_rectangle(0, h - 20, w, h, fill="", outline="", stipple="gray75")
        
        self.flicker = random.random()
        self._after_id = self.after(80, self._draw_scanlines)

# ============================================================================
# NEON GLOW BUTTON
# ============================================================================
class NeonButton(tk.Canvas):
    """
    Sci-fi button with neon glow border and hover effects.
    Real canvas-drawn button with click handling.
    """
    
    def __init__(self, parent, text="BUTTON", width=120, height=36,
                 color=Colors.CRT_CYAN, hover_color=Colors.ATLANTEAN_TEAL,
                 command=None, **kwargs):
        super().__init__(parent, width=width, height=height, bg=Colors.DARK_PANEL,
                        highlightthickness=0, **kwargs)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.command = command
        self.hovered = False
        self.pressed = False
        self._draw()
        
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
    
    def _draw(self):
        self.delete("all")
        w = int(self['width'])
        h = int(self['height'])
        
        color = self.hover_color if self.hovered else self.color
        if self.pressed:
            color = Colors.SACRED_GOLD
        
        # Outer glow
        for i in range(3, 0, -1):
            alpha = 30 - i * 8
            glow = f"#{0:02x}{alpha + 0x40:02x}{alpha + 0x40:02x}"
            self.create_rectangle(i, i, w - i, h - i,
                                outline=glow, width=1)
        
        # Main border
        self.create_rectangle(2, 2, w - 2, h - 2,
                            outline=color, width=2)
        
        # Corner accents
        corner = 6
        self.create_line(0, 0, corner, 0, fill=color, width=2)
        self.create_line(0, 0, 0, corner, fill=color, width=2)
        self.create_line(w, 0, w - corner, 0, fill=color, width=2)
        self.create_line(w, 0, w, corner, fill=color, width=2)
        self.create_line(0, h, corner, h, fill=color, width=2)
        self.create_line(0, h, 0, h - corner, fill=color, width=2)
        self.create_line(w, h, w - corner, h, fill=color, width=2)
        self.create_line(w, h, w, h - corner, fill=color, width=2)
        
        # Text
        self.create_text(w // 2, h // 2, text=self.text,
                        fill=color, font=("Consolas", 10, "bold"))
    
    def _on_enter(self, event):
        self.hovered = True
        self._draw()
    
    def _on_leave(self, event):
        self.hovered = False
        self.pressed = False
        self._draw()
    
    def _on_press(self, event):
        self.pressed = True
        self._draw()
    
    def _on_release(self, event):
        self.pressed = False
        self._draw()
        if self.command and self.hovered:
            self.command()

# ============================================================================
# CRYSTALLINE PANEL FRAME
# ============================================================================
class CrystallineFrame(tk.Canvas):
    """
    Panel with Atlantean crystalline border design.
    Hexagonal corner accents with teal glow.
    """
    
    def __init__(self, parent, width=300, height=200, title="PANEL",
                 color=Colors.CRYSTAL_BLUE, **kwargs):
        super().__init__(parent, width=width, height=height, bg=Colors.DARK_PANEL,
                        highlightthickness=0, **kwargs)
        self.title = title
        self.color = color
        self._draw()
    
    def _draw(self):
        self.delete("all")
        w = int(self['width'])
        h = int(self['height'])
        
        # Background gradient effect (simulated with rectangles)
        for i in range(h // 4):
            shade = max(0, 0x11 - i // 4)
            color = f"#{shade:02x}{shade + 0x10:02x}{shade + 0x20:02x}"
            self.create_rectangle(0, i * 4, w, (i + 1) * 4,
                                fill=color, outline="")
        
        # Main border
        self.create_rectangle(4, 4, w - 4, h - 4, outline=self.color, width=1)
        
        # Title bar
        self.create_rectangle(4, 4, w - 4, 24, fill=self.color, outline="", stipple="gray25")
        self.create_text(10, 14, text=f"◆ {self.title}", fill=Colors.ATLANTEAN_TEAL,
                        font=("Consolas", 9, "bold"), anchor="w")
        
        # Hexagonal corner crystals
        hex_size = 8
        corners = [(4, 4), (w - 4, 4), (4, h - 4), (w - 4, h - 4)]
        for cx, cy in corners:
            points = []
            for i in range(6):
                angle = math.radians(i * 60)
                px = cx + math.cos(angle) * hex_size
                py = cy + math.sin(angle) * hex_size
                points.extend([px, py])
            self.create_polygon(points, fill=self.color, outline=Colors.AQUA_GLOW, width=1)
        
        # Side accents
        self.create_line(20, h - 4, w - 20, h - 4, fill=self.color, width=1)

# ============================================================================
# TERMINAL LOG PANEL
# ============================================================================
class TerminalLog(tk.Text):
    """
    Terminal-style scrolling log with green text, typing effect,
    and Matrix-style character reveal for new lines.
    """
    
    def __init__(self, parent, height=10, width=50, **kwargs):
        super().__init__(parent, height=height, width=width, bg=Colors.VOID_BLACK,
                        fg=Colors.MATRIX_GREEN, font=("Consolas", 9),
                        wrap=tk.WORD, state=tk.DISABLED, **kwargs)
        self.tag_configure("timestamp", foreground=Colors.CRT_CYAN)
        self.tag_configure("warning", foreground=Colors.WARNING)
        self.tag_configure("alert", foreground=Colors.ALERT)
        self.tag_configure("ok", foreground=Colors.OK)
        self.tag_configure("system", foreground=Colors.ATLANTEAN_TEAL)
        self.max_lines = 100
        
    def log(self, message, level="info"):
        """Add a log line with timestamp and color coding."""
        self.config(state=tk.NORMAL)
        
        ts = time.strftime("%H:%M:%S")
        tag = {
            "warning": "warning",
            "alert": "alert",
            "ok": "ok",
            "system": "system",
        }.get(level, "")
        
        line = f"[{ts}] {message}\n"
        
        if tag:
            self.insert(tk.END, line, tag)
        else:
            self.insert(tk.END, line)
        
        # Keep max lines
        lines = int(self.index('end-1c').split('.')[0])
        if lines > self.max_lines:
            self.delete('1.0', f'{lines - self.max_lines}.0')
        
        self.see(tk.END)
        self.config(state=tk.DISABLED)
    
    def clear(self):
        self.config(state=tk.NORMAL)
        self.delete('1.0', tk.END)
        self.config(state=tk.DISABLED)

# ============================================================================
# SCI-FI HUD CROSSHAIR & DATA RINGS
# ============================================================================
class HUDOverlay(tk.Canvas):
    """
    Real-time sci-fi HUD overlay:
    - Animated targeting reticle
    - Rotating data rings
    - Pulse indicators
    - Corner brackets
    """
    
    def __init__(self, parent, width=640, height=480, **kwargs):
        super().__init__(parent, width=width, height=height, bg="",
                        highlightthickness=0, **kwargs)
        self['bg'] = ''
        self.rotation = 0
        self.running = False
        self._after_id = None
        self.target_x = width // 2
        self.target_y = height // 2
        
    def start(self):
        self.running = True
        self._animate()
    
    def stop(self):
        self.running = False
        if self._after_id:
            self.after_cancel(self._after_id)
    
    def set_target(self, x, y):
        self.target_x = x
        self.target_y = y
    
    def _animate(self):
        if not self.running:
            return
        
        self.delete("all")
        w = int(self['width'])
        h = int(self['height'])
        cx, cy = self.target_x, self.target_y
        self.rotation += 2
        
        # Outer data ring (dashed)
        outer_r = 80
        segments = 24
        for i in range(segments):
            angle_start = math.radians(self.rotation + i * (360 / segments))
            angle_end = math.radians(self.rotation + i * (360 / segments) + 5)
            self.create_arc(cx - outer_r, cy - outer_r, cx + outer_r, cy + outer_r,
                          start=i * (360 / segments) + self.rotation,
                          extent=5, outline=Colors.CRYSTAL_BLUE, width=1, style="arc")
        
        # Inner rotating ring
        inner_r = 50
        self.create_oval(cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r,
                       outline=Colors.ATLANTEAN_TEAL, width=2)
        
        # Crosshair
        ch_len = 20
        self.create_line(cx - ch_len - 10, cy, cx - 10, cy,
                       fill=Colors.CRT_CYAN, width=2)
        self.create_line(cx + 10, cy, cx + ch_len + 10, cy,
                       fill=Colors.CRT_CYAN, width=2)
        self.create_line(cx, cy - ch_len - 10, cx, cy - 10,
                       fill=Colors.CRT_CYAN, width=2)
        self.create_line(cx, cy + 10, cx, cy + ch_len + 10,
                       fill=Colors.CRT_CYAN, width=2)
        
        # Center dot with pulse
        pulse = 4 + int((math.sin(time.time() * 4) + 1) * 3)
        self.create_oval(cx - pulse, cy - pulse, cx + pulse, cy + pulse,
                       fill=Colors.ALERT, outline=Colors.WARNING, width=1)
        
        # Corner brackets (screen corners)
        bracket = 30
        corners = [(0, 0), (w, 0), (0, h), (w, h)]
        for i, (corner_x, corner_y) in enumerate(corners):
            dx = 1 if corner_x == 0 else -1
            dy = 1 if corner_y == 0 else -1
            self.create_line(corner_x, corner_y, corner_x + bracket * dx, corner_y,
                           fill=Colors.CRYSTAL_BLUE, width=2)
            self.create_line(corner_x, corner_y, corner_x, corner_y + bracket * dy,
                           fill=Colors.CRYSTAL_BLUE, width=2)
        
        # Side bars (signal strength style)
        for i in range(5):
            bar_h = 5 + i * 6
            bx = w - 25
            by = h - 20 - bar_h
            color = Colors.MATRIX_GREEN if i < 4 else Colors.WARNING
            self.create_rectangle(bx + i * 5, by, bx + i * 5 + 3, h - 20,
                                fill=color, outline="")
        
        self._after_id = self.after(50, self._animate)

# ============================================================================
# GLITCH TEXT LABEL
# ============================================================================
class GlitchLabel(tk.Label):
    """
    Label that occasionally glitches its text with character substitution
    and color shifts for a corrupted digital effect.
    """
    
    GLITCH_CHARS = "@#$%&*!?<>[]{}|~"
    
    def __init__(self, parent, text="", color=Colors.CRT_CYAN, **kwargs):
        font = kwargs.pop("font", ("Consolas", 12, "bold"))
        super().__init__(parent, text=text, fg=color, bg=Colors.DARK_PANEL,
                        font=font, **kwargs)
        self._original_text = text
        self._color = color
        self._glitching = False
        self._after_id = None
        self._start_glitch_loop()
    
    def _start_glitch_loop(self):
        if random.random() < 0.03:  # 3% chance per cycle
            self._glitch()
        self._after_id = self.after(200, self._start_glitch_loop)
    
    def _glitch(self):
        if self._glitching:
            return
        self._glitching = True
        
        # Scramble some characters
        chars = list(self._original_text)
        indices = random.sample(range(len(chars)), min(3, len(chars)))
        original = [chars[i] for i in indices]
        
        for i in indices:
            if chars[i] != ' ':
                chars[i] = random.choice(self.GLITCH_CHARS)
        
        self.config(text=''.join(chars), fg=Colors.NEON_PINK)
        
        # Restore after brief moment
        self.after(80, lambda: self._restore(original, indices))
    
    def _restore(self, original, indices):
        chars = list(self._original_text)
        for i, orig in zip(indices, original):
            chars[i] = orig
        self.config(text=''.join(chars), fg=self._color)
        self._glitching = False
    
    def set_text(self, text):
        self._original_text = text
        if not self._glitching:
            self.config(text=text)
    
    def destroy(self):
        if self._after_id:
            self.after_cancel(self._after_id)
        super().destroy()

# ============================================================================
# PROGRESS BAR - Crystalline style
# ============================================================================
class CrystalProgressBar(tk.Canvas):
    """
    Horizontal progress bar with crystalline segments and glow.
    """
    
    def __init__(self, parent, width=200, height=16, color=Colors.ATLANTEAN_TEAL, **kwargs):
        super().__init__(parent, width=width, height=height, bg=Colors.DARK_PANEL,
                        highlightthickness=0, **kwargs)
        self.color = color
        self.value = 0.0  # 0.0 to 1.0
        self._draw()
    
    def set_value(self, value):
        self.value = max(0.0, min(1.0, value))
        self._draw()
    
    def _draw(self):
        self.delete("all")
        w = int(self['width'])
        h = int(self['height'])
        
        # Background
        self.create_rectangle(1, 1, w - 1, h - 1, outline=self.color, width=1)
        
        # Fill with segments
        fill_w = int((w - 4) * self.value)
        if fill_w > 0:
            segments = max(1, fill_w // 8)
            seg_w = fill_w / segments
            for i in range(segments):
                x1 = 2 + i * seg_w
                x2 = 2 + (i + 1) * seg_w - 1
                alpha = 0x88 + int(0x77 * (i / segments))
                color = f"#00{alpha:02x}{alpha:02x}"
                self.create_rectangle(x1, 3, x2, h - 3, fill=color, outline="")
        
        # Percentage text
        self.create_text(w // 2, h // 2, text=f"{int(self.value * 100)}%",
                        fill=Colors.CRT_CYAN, font=("Consolas", 8))

# ============================================================================
# HEX DATA DISPLAY
# ============================================================================
class HexDisplay(tk.Canvas):
    """
    Scrolling hex dump display like a memory viewer.
    Shows random hex values that update periodically.
    """
    
    def __init__(self, parent, rows=8, cols=8, width=200, height=160, **kwargs):
        super().__init__(parent, width=width, height=height, bg=Colors.VOID_BLACK,
                        highlightthickness=0, **kwargs)
        self.rows = rows
        self.cols = cols
        self.values = [[random.randint(0, 255) for _ in range(cols)] for _ in range(rows)]
        self.running = False
        self._after_id = None
        self._draw()
    
    def start(self):
        self.running = True
        self._update()
    
    def stop(self):
        self.running = False
        if self._after_id:
            self.after_cancel(self._after_id)
    
    def _update(self):
        if not self.running:
            return
        # Randomly change some values
        for _ in range(3):
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            self.values[r][c] = random.randint(0, 255)
        self._draw()
        self._after_id = self.after(150, self._update)
    
    def _draw(self):
        self.delete("all")
        w = int(self['width'])
        h = int(self['height'])
        
        cell_w = w / self.cols
        cell_h = h / self.rows
        
        for r in range(self.rows):
            for c in range(self.cols):
                x = c * cell_w + cell_w / 2
                y = r * cell_h + cell_h / 2
                val = self.values[r][c]
                
                # Color based on value
                if val > 200:
                    color = Colors.WARNING
                elif val > 150:
                    color = Colors.ATLANTEAN_TEAL
                else:
                    color = Colors.MATRIX_DARK
                
                self.create_text(x, y, text=f"{val:02X}",
                               fill=color, font=("Consolas", 8))

# ============================================================================
# WAVEFORM VISUALIZER
# ============================================================================
class WaveformCanvas(tk.Canvas):
    """
    Animated waveform/bar visualizer like an audio spectrum or data stream.
    """
    
    def __init__(self, parent, bars=32, width=300, height=60, color=Colors.CRYSTAL_BLUE, **kwargs):
        super().__init__(parent, width=width, height=height, bg=Colors.VOID_BLACK,
                        highlightthickness=0, **kwargs)
        self.bars = bars
        self.bar_color = color
        self.values = [0.0] * bars
        self.running = False
        self._after_id = None
    
    def start(self):
        self.running = True
        self._animate()
    
    def stop(self):
        self.running = False
        if self._after_id:
            self.after_cancel(self._after_id)
    
    def _animate(self):
        if not self.running:
            return
        
        # Update values with Perlin-ish noise
        for i in range(self.bars):
            target = (math.sin(time.time() * 3 + i * 0.5) + 1) / 2
            self.values[i] = self.values[i] * 0.7 + target * 0.3
        
        self.delete("all")
        w = int(self['width'])
        h = int(self['height'])
        bar_w = w / self.bars
        
        for i, val in enumerate(self.values):
            x1 = i * bar_w + 1
            x2 = (i + 1) * bar_w - 1
            bar_h = val * (h - 4)
            y1 = h - 2 - bar_h
            y2 = h - 2
            
            # Color gradient from blue to cyan
            g = int(0x44 + val * 0xBB)
            color = f"#00{g:02x}{g:02x}"
            self.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        
        self._after_id = self.after(50, self._animate)

# ============================================================================
# STATUS INDICATOR - LED-style status light with pulse
# ============================================================================
class StatusIndicator(tk.Canvas):
    """
    LED-style status indicator with pulsing glow animation.
    Shows real system state with color coding.
    """
    
    def __init__(self, parent, width=16, height=16, color=Colors.ALERT, **kwargs):
        super().__init__(parent, width=width, height=height, bg=Colors.DARK_PANEL,
                        highlightthickness=0, **kwargs)
        self.color = color
        self.pulse = 0.0
        self._after_id = None
        self._animate()
    
    def set_color(self, color):
        """Change indicator color."""
        self.color = color
        self._draw()
    
    def _animate(self):
        self.pulse = (math.sin(time.time() * 3) + 1) / 2
        self._draw()
        self._after_id = self.after(100, self._animate)
    
    def _draw(self):
        self.delete("all")
        w = int(self['width'])
        h = int(self['height'])
        cx, cy = w // 2, h // 2
        r = min(cx, cy) - 2
        
        # Outer glow (pulsing)
        glow_r = r + 2 + int(self.pulse * 3)
        glow_alpha = int(30 + self.pulse * 40)
        glow_color = self._hex_with_alpha(self.color, glow_alpha)
        self.create_oval(cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r,
                        fill=glow_color, outline="")
        
        # Main circle
        self.create_oval(cx - r, cy - r, cx + r, cy + r,
                        fill=self.color, outline="")
        
        # Highlight
        self.create_oval(cx - r//2, cy - r//2, cx + r//3, cy + r//3,
                        fill="white", outline="", stipple="gray75")
    
    def _hex_with_alpha(self, hex_color, alpha):
        """Convert hex color to RGB tuple for stipple."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def destroy(self):
        if self._after_id:
            self.after_cancel(self._after_id)
        super().destroy()

# ============================================================================
# BLENDSHAPE BARS - Real-time blendshape visualization
# ============================================================================
class BlendshapeBars(tk.Canvas):
    """
    Real-time horizontal bar chart for blendshape values.
    Updates from actual tracking data.
    """
    
    def __init__(self, parent, width=260, height=200, max_bars=12, **kwargs):
        super().__init__(parent, width=width, height=height, bg=Colors.DARK_PANEL,
                        highlightthickness=0, **kwargs)
        self.max_bars = max_bars
        self.bar_height = 14
        self.bar_gap = 4
        self.values = {}  # name -> value
        self._draw_empty()
    
    def update_values(self, values: dict):
        """Update blendshape values. values: dict of name->float (0.0-1.0)"""
        self.values = values
        self._draw()
    
    def _draw_empty(self):
        self.delete("all")
        w = int(self['width'])
        h = int(self['height'])
        self.create_text(w // 2, h // 2, text="AWAITING TRACKING DATA...",
                        fill=Colors.MATRIX_DARK, font=("Consolas", 10))
    
    def _draw(self):
        self.delete("all")
        w = int(self['width'])
        
        # Sort by value descending, take top max_bars
        sorted_vals = sorted(self.values.items(), key=lambda x: x[1], reverse=True)[:self.max_bars]
        
        y_offset = 2
        for name, val in sorted_vals:
            bar_w = int(val * (w - 80))
            
            # Background bar
            self.create_rectangle(75, y_offset, w - 5, y_offset + self.bar_height,
                                outline=Colors.SCANLINE, width=1)
            
            # Fill bar with gradient color
            if val > 0.7:
                color = Colors.WARNING
            elif val > 0.4:
                color = Colors.ATLANTEAN_TEAL
            else:
                color = Colors.CRYSTAL_BLUE
            
            if bar_w > 0:
                self.create_rectangle(75, y_offset, 75 + bar_w, y_offset + self.bar_height,
                                    fill=color, outline="", stipple="gray75")
            
            # Label (truncated)
            label = name[:12].ljust(12)
            self.create_text(2, y_offset + self.bar_height // 2, text=label,
                           fill=Colors.CRT_CYAN, font=("Consolas", 8), anchor="w")
            
            # Percentage
            pct = f"{int(val * 100):3d}%"
            self.create_text(w - 2, y_offset + self.bar_height // 2, text=pct,
                           fill=Colors.CRT_GREEN, font=("Consolas", 8), anchor="e")
            
            y_offset += self.bar_height + self.bar_gap

# ============================================================================
# THEME HELPER - Apply theme to standard widgets
# ============================================================================
def apply_dark_theme(root):
    """Apply dark sci-fi theme to tkinter root window."""
    root.configure(bg=Colors.DEEP_SPACE)
    
    style = ttk.Style()
    style.theme_use('clam')
    
    style.configure("TFrame", background=Colors.DARK_PANEL)
    style.configure("TLabel", background=Colors.DARK_PANEL, foreground=Colors.CRT_CYAN,
                   font=("Consolas", 10))
    style.configure("TButton", background=Colors.DARK_PANEL, foreground=Colors.CRT_CYAN,
                   font=("Consolas", 10, "bold"))
    style.configure("TScale", background=Colors.DARK_PANEL, troughcolor=Colors.SCANLINE)
    style.configure("TCheckbutton", background=Colors.DARK_PANEL, foreground=Colors.CRT_CYAN,
                   font=("Consolas", 9))
    style.configure("TEntry", fieldbackground=Colors.VOID_BLACK, foreground=Colors.CRT_CYAN,
                   insertcolor=Colors.CRT_CYAN)

# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("GUI Themes Test")
    root.configure(bg=Colors.DEEP_SPACE)
    apply_dark_theme(root)
    
    # Matrix rain
    rain = MatrixRainCanvas(root, width=400, height=200, column_spacing=12)
    rain.pack(pady=5)
    rain.start()
    
    # Sacred geometry
    geo = SacredGeometryCanvas(root, width=200, height=200)
    geo.pack(pady=5)
    geo.start()
    
    # Neon button
    def on_click():
        print("Neon button clicked!")
    
    btn = NeonButton(root, text="ACTIVATE", command=on_click)
    btn.pack(pady=5)
    
    # Crystalline panel
    panel = CrystallineFrame(root, width=300, height=100, title="SYSTEM STATUS")
    panel.pack(pady=5)
    
    # Terminal log
    log = TerminalLog(root, height=5, width=50)
    log.pack(pady=5)
    log.log("System initialized", "system")
    log.log("Matrix rain active", "ok")
    log.log("Connection established", "ok")
    
    # Hex display
    hex_disp = HexDisplay(root, rows=4, cols=6, width=180, height=80)
    hex_disp.pack(pady=5)
    hex_disp.start()
    
    # Waveform
    wave = WaveformCanvas(root, bars=24, width=300, height=50)
    wave.pack(pady=5)
    wave.start()
    
    # Glitch label
    glitch = GlitchLabel(root, text="MAX HEADROOM DIGITIZER", color=Colors.CRT_CYAN)
    glitch.pack(pady=5)
    
    # Progress bar
    prog = CrystalProgressBar(root, width=200, height=16)
    prog.pack(pady=5)
    prog.set_value(0.75)
    
    root.mainloop()
