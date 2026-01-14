import pygame
import sys
import math
import tkinter as tk
from tkinter import filedialog
from pygame.locals import *

pygame.init()
root = tk.Tk()
root.withdraw()

# ================= НАСТРОЙКИ =================
WIDTH, HEIGHT = 1100, 750
FPS = 60

MODE_NONE = 0
MODE_RECT = 10
MODE_CIRCLE = 11
MODE_POLYGON = 12
MODE_RECT_CIRCLE_RECT = 13
MODE_ARROW = 14

FILL_COLOR = 1
FILL_TEXTURE = 2

# ================= ОКНО =================
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Painter – MFC style (Python/Pygame)")
clock = pygame.time.Clock()

font = pygame.font.SysFont("Segoe UI", 16)

# ================= СОСТОЯНИЕ =================
current_mode = MODE_NONE
current_fill_mode = FILL_COLOR
current_color = (0, 0, 0)
current_outline_color = (0, 0, 0)
current_outline_width = 2
current_texture = None

background_image = None

start_pos = None
polygon_points = []

shapes = []

# ================= УТИЛИТЫ =================

def draw_text(text, x, y):
    img = font.render(text, True, (0, 0, 0))
    screen.blit(img, (x, y))

def open_file_dialog():
    return filedialog.askopenfilename(filetypes=[("Bitmap files", "*.bmp")])

def new_file():
    global shapes, background_image
    shapes = []
    background_image = None

def save_bmp():
    surface = pygame.Surface((WIDTH, HEIGHT))
    surface.fill((255, 255, 255))
    if background_image:
        surface.blit(background_image, (0, 0))
    for shape in shapes:
        shape.draw(surface)
    pygame.image.save(surface, "output.bmp")
    print("Сохранено: output.bmp")

def load_background():
    global background_image
    path = open_file_dialog()
    if not path:
        return
    img = pygame.image.load(path).convert()
    background_image = pygame.transform.scale(img, (WIDTH, HEIGHT))

def load_texture():
    global current_texture
    path = open_file_dialog()
    if not path:
        return
    current_texture = pygame.image.load(path).convert_alpha()

# ================= ЗАЛИВКА ТЕКСТУРОЙ =================

def create_tiled_surface(size, texture):
    w, h = size
    surf = pygame.Surface((w, h), SRCALPHA)
    tw, th = texture.get_size()
    for y in range(0, h, th):
        for x in range(0, w, tw):
            surf.blit(texture, (x, y))
    return surf

def fill_polygon_with_texture(target, points, texture):
    if not texture:
        return
    mask = pygame.Surface((WIDTH, HEIGHT), SRCALPHA)
    pygame.draw.polygon(mask, (255, 255, 255, 255), points)
    tiled = create_tiled_surface((WIDTH, HEIGHT), texture)
    tiled.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    target.blit(tiled, (0, 0))

def fill_circle_with_texture(target, center, radius, texture):
    if not texture:
        return
    mask = pygame.Surface((WIDTH, HEIGHT), SRCALPHA)
    pygame.draw.circle(mask, (255, 255, 255, 255), center, radius)
    tiled = create_tiled_surface((WIDTH, HEIGHT), texture)
    tiled.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    target.blit(tiled, (0, 0))

def fill_rect_with_texture(target, rect, texture):
    points = [
        (rect.left, rect.top),
        (rect.right, rect.top),
        (rect.right, rect.bottom),
        (rect.left, rect.bottom)
    ]
    fill_polygon_with_texture(target, points, texture)

# ================= БАЗОВЫЙ КЛАСС =================

class Shape:
    def __init__(self, fill_mode, color, outline_color, outline_width, texture):
        self.fill_mode = fill_mode
        self.color = color
        self.outline_color = outline_color
        self.outline_width = outline_width
        self.texture = texture

    def draw(self, surface):
        pass

    def move(self, dx, dy):
        pass

    def rotate(self, angle):
        pass

# ================= ФИГУРЫ =================

class RectShape(Shape):
    def __init__(self, rect, **kwargs):
        super().__init__(**kwargs)
        self.rect = rect

    def draw(self, surface):
        if self.fill_mode == FILL_COLOR:
            pygame.draw.rect(surface, self.color, self.rect)
        else:
            fill_rect_with_texture(surface, self.rect, self.texture)
        pygame.draw.rect(surface, self.outline_color, self.rect, self.outline_width)

    def move(self, dx, dy):
        self.rect.move_ip(dx, dy)

class CircleShape(Shape):
    def __init__(self, center, radius, **kwargs):
        super().__init__(**kwargs)
        self.center = list(center)
        self.radius = radius

    def draw(self, surface):
        if self.fill_mode == FILL_COLOR:
            pygame.draw.circle(surface, self.color, self.center, self.radius)
        else:
            fill_circle_with_texture(surface, self.center, self.radius, self.texture)
        pygame.draw.circle(surface, self.outline_color, self.center, self.radius, self.outline_width)

    def move(self, dx, dy):
        self.center[0] += dx
        self.center[1] += dy

class PolygonShape(Shape):
    def __init__(self, points, **kwargs):
        super().__init__(**kwargs)
        self.points = points

    def draw(self, surface):
        if self.fill_mode == FILL_COLOR:
            pygame.draw.polygon(surface, self.color, self.points)
        else:
            fill_polygon_with_texture(surface, self.points, self.texture)
        pygame.draw.polygon(surface, self.outline_color, self.points, self.outline_width)

    def move(self, dx, dy):
        self.points = [(x + dx, y + dy) for x, y in self.points]

    def rotate(self, angle_deg):
        angle = math.radians(angle_deg)
        cx = sum(p[0] for p in self.points) / len(self.points)
        cy = sum(p[1] for p in self.points) / len(self.points)

        new_points = []
        for x, y in self.points:
            tx, ty = x - cx, y - cy
            rx = tx * math.cos(angle) - ty * math.sin(angle)
            ry = tx * math.sin(angle) + ty * math.cos(angle)
            new_points.append((rx + cx, ry + cy))
        self.points = new_points

class RectCircleRectShape(Shape):
    def __init__(self, p1, p2, **kwargs):
        super().__init__(**kwargs)

        x1, y1 = p1
        x2, y2 = p2

        cx = (x1 + x2) // 2
        top = min(y1, y2)
        height = abs(y2 - y1)
        width = abs(x2 - x1)

        rect_h = height // 4
        circle_r = width // 2
        gap = height // 8

        self.top_rect = pygame.Rect(cx - circle_r, top, 2 * circle_r, rect_h)
        self.bottom_rect = pygame.Rect(cx - circle_r, top + rect_h + 2 * circle_r, 2 * circle_r, rect_h)
        self.center = [cx, top + rect_h + circle_r]
        self.circle_r = circle_r

    def draw(self, surface):
        if self.fill_mode == FILL_COLOR:
            pygame.draw.rect(surface, self.color, self.top_rect)
            pygame.draw.rect(surface, self.color, self.bottom_rect)
            pygame.draw.circle(surface, self.color, self.center, self.circle_r)
        else:
            fill_rect_with_texture(surface, self.top_rect, self.texture)
            fill_rect_with_texture(surface, self.bottom_rect, self.texture)
            fill_circle_with_texture(surface, self.center, self.circle_r, self.texture)

        pygame.draw.rect(surface, self.outline_color, self.top_rect, self.outline_width)
        pygame.draw.rect(surface, self.outline_color, self.bottom_rect, self.outline_width)
        pygame.draw.circle(surface, self.outline_color, self.center, self.circle_r, self.outline_width)

    def move(self, dx, dy):
        self.top_rect.move_ip(dx, dy)
        self.bottom_rect.move_ip(dx, dy)
        self.center[0] += dx
        self.center[1] += dy

class ArrowShape(PolygonShape):
    def __init__(self, p1, p2, **kwargs):
        x1, y1 = p1
        x2, y2 = p2

        left = min(x1, x2)
        right = max(x1, x2)
        top = min(y1, y2)
        bottom = max(y1, y2)

        w = right - left
        h = bottom - top
        mid_y = top + h / 2

        points = [
            (left, top),
            (left + w * 0.7, top),
            (left + w * 0.7, top),
            (left + w * 0.7, top - h * 0.3),
            (right, mid_y),
            (left + w * 0.7, bottom + h * 0.3),
            (left + w * 0.7, bottom),
            (left, bottom)
        ]

        super().__init__(points, **kwargs)

# ================= ГЛАВНЫЙ ЦИКЛ =================

running = True
while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

        elif event.type == KEYDOWN:

            # --- режимы ---
            if event.key == K_q:
                current_mode = MODE_RECT
            elif event.key == K_w:
                current_mode = MODE_CIRCLE
            elif event.key == K_e:
                current_mode = MODE_POLYGON
                polygon_points = []
            elif event.key == K_r:
                current_mode = MODE_RECT_CIRCLE_RECT
            elif event.key == K_t:
                current_mode = MODE_ARROW
            elif event.key == K_ESCAPE:
                current_mode = MODE_NONE

            # --- файл ---
            elif event.key == K_n:
                new_file()
            elif event.key == K_p:
                save_bmp()
            elif event.key == K_o:
                load_background()
            elif event.key == K_l:
                load_texture()

            # --- заливка ---
            elif event.key == K_f:
                current_fill_mode = FILL_COLOR
            elif event.key == K_g:
                current_fill_mode = FILL_TEXTURE

            # --- цвета ---
            elif event.key == K_1:
                current_color = (0, 0, 0)
            elif event.key == K_2:
                current_color = (255, 0, 0)
            elif event.key == K_3:
                current_color = (0, 0, 255)
            elif event.key == K_4:
                current_color = (255, 255, 0)

            # --- завершить полигон ---
            elif event.key == K_RETURN:
                if current_mode == MODE_POLYGON and len(polygon_points) >= 3:
                    shapes.append(
                        PolygonShape(
                            polygon_points.copy(),
                            fill_mode=current_fill_mode,
                            color=current_color,
                            outline_color=current_outline_color,
                            outline_width=current_outline_width,
                            texture=current_texture
                        )
                    )
                    polygon_points = []

            # --- трансформации (ТОЛЬКО стрелки + Q/E) ---
            elif event.key == K_LEFT and shapes:
                shapes[-1].move(-10, 0)
            elif event.key == K_RIGHT and shapes:
                shapes[-1].move(10, 0)
            elif event.key == K_UP and shapes:
                shapes[-1].move(0, -10)
            elif event.key == K_DOWN and shapes:
                shapes[-1].move(0, 10)
            elif event.key == K_a and shapes:
                shapes[-1].rotate(-10)
            elif event.key == K_d and shapes:
                shapes[-1].rotate(10)

        elif event.type == MOUSEBUTTONDOWN and event.button == 1:
            start_pos = event.pos
            if current_mode == MODE_POLYGON:
                polygon_points.append(event.pos)

        elif event.type == MOUSEBUTTONUP and event.button == 1 and start_pos:
            if current_mode == MODE_RECT:
                x1, y1 = start_pos
                x2, y2 = event.pos
                rect = pygame.Rect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
                shapes.append(RectShape(rect,
                                        fill_mode=current_fill_mode,
                                        color=current_color,
                                        outline_color=current_outline_color,
                                        outline_width=current_outline_width,
                                        texture=current_texture))

            elif current_mode == MODE_CIRCLE:
                r = int(math.hypot(event.pos[0] - start_pos[0], event.pos[1] - start_pos[1]))
                shapes.append(CircleShape(start_pos, r,
                                          fill_mode=current_fill_mode,
                                          color=current_color,
                                          outline_color=current_outline_color,
                                          outline_width=current_outline_width,
                                          texture=current_texture))

            elif current_mode == MODE_RECT_CIRCLE_RECT:
                shapes.append(RectCircleRectShape(start_pos, event.pos,
                                                  fill_mode=current_fill_mode,
                                                  color=current_color,
                                                  outline_color=current_outline_color,
                                                  outline_width=current_outline_width,
                                                  texture=current_texture))

            elif current_mode == MODE_ARROW:
                shapes.append(ArrowShape(start_pos, event.pos,
                                         fill_mode=current_fill_mode,
                                         color=current_color,
                                         outline_color=current_outline_color,
                                         outline_width=current_outline_width,
                                         texture=current_texture))

            start_pos = None

    # ============ ОТРИСОВКА ============
    screen.fill((230, 230, 230))

    if background_image:
        screen.blit(background_image, (0, 0))

    for shape in shapes:
        shape.draw(screen)

    if current_mode == MODE_POLYGON and len(polygon_points) > 1:
        pygame.draw.lines(screen, (0, 0, 0), False, polygon_points, 2)

    draw_text("Q-Прямоуг  W-Круг  E-Многоуг  R-[|O|]  T-Стрелка", 10, 10)
    draw_text("1-Чёрн 2-Красн 3-Синий 4-Жёлт   F-Цвет  G-Текстура", 10, 30)
    draw_text("N-Новый  P-Сохранить  O-Фон  L-Текстура", 10, 50)
    draw_text("Стрелки-Перемещение  A/D-Поворот последней фигуры", 10, 70)

    pygame.display.flip()

pygame.quit()
sys.exit()
