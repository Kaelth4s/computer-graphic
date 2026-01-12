import pygame
import sys
import os
import numpy as np
from pygame.locals import *
from scipy import ndimage  # Для быстрой бикубической интерполяции

# Инициализация Pygame
pygame.init()

# Константы
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
BACKGROUND_COLOR = (40, 44, 52)
PANEL_COLOR = (30, 34, 42)
TEXT_COLOR = (220, 220, 220)
HIGHLIGHT_COLOR = (86, 182, 194)


class BMPLoader:
    """Класс для загрузки и обработки BMP изображений"""

    def __init__(self):
        self.image = None
        self.original_surface = None
        self.scaled_surface = None
        self.filename = ""
        self.original_array = None

    def load_bmp(self, filename):
        """Загрузка BMP файла"""
        try:
            self.original_surface = pygame.image.load(filename)
            self.image = self.original_surface
            self.filename = os.path.basename(filename)
            self.original_array = pygame.surfarray.array3d(self.original_surface)
            return True
        except pygame.error as e:
            print(f"Ошибка загрузки изображения: {e}")
            return False

    def nearest_neighbor_scale(self, scale_factor):
        """Масштабирование методом ближайшего соседа (оптимизированная версия)"""
        if self.original_array is None:
            return

        original_height, original_width = self.original_array.shape[:2]

        new_width = max(1, int(original_width * scale_factor))
        new_height = max(1, int(original_height * scale_factor))

        # Используем встроенную функцию Pygame для скорости
        self.scaled_surface = pygame.transform.scale(self.original_surface, (new_width, new_height))
        self.image = self.scaled_surface

    def linear_interpolation_scale(self, scale_factor):
        """Масштабирование методом линейной интерполяции (оптимизированная версия)"""
        if self.original_array is None:
            return

        original_height, original_width = self.original_array.shape[:2]

        new_width = max(1, int(original_width * scale_factor))
        new_height = max(1, int(original_height * scale_factor))

        # Используем smoothscale Pygame для билинейной интерполяции
        self.scaled_surface = pygame.transform.smoothscale(self.original_surface, (new_width, new_height))
        self.image = self.scaled_surface

    def bicubic_interpolation_scale(self, scale_factor):
        """Масштабирование методом бикубической интерполяции (оптимизированная версия)"""
        if self.original_array is None:
            return

        original_height, original_width = self.original_array.shape[:2]

        new_width = max(1, int(original_width * scale_factor))
        new_height = max(1, int(original_height * scale_factor))

        # Используем scipy для быстрой бикубической интерполяции
        scaled_array = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        for channel in range(3):
            # Масштабируем каждый канал отдельно
            scaled_channel = ndimage.zoom(self.original_array[:, :, channel],
                                          scale_factor,
                                          order=3)  # order=3 для бикубической интерполяции

            # Обрезаем до нужного размера (zoom может дать немного другой размер)
            h, w = scaled_channel.shape
            if h > new_height or w > new_width:
                scaled_channel = scaled_channel[:new_height, :new_width]

            # Заполняем выходной массив
            target_h, target_w = min(h, new_height), min(w, new_width)
            scaled_array[:target_h, :target_w, channel] = scaled_channel[:target_h, :target_w]

        self.scaled_surface = pygame.surfarray.make_surface(scaled_array)
        self.image = self.scaled_surface

    def apply_scaling(self, scale_factor, mode):
        """Применение масштабирования с указанным режимом"""
        if self.original_surface is None:
            return

        if abs(scale_factor - 1.0) < 0.01:
            self.image = self.original_surface
            self.scaled_surface = None
            return

        if mode == 0:
            self.nearest_neighbor_scale(scale_factor)
        elif mode == 1:
            self.linear_interpolation_scale(scale_factor)
        elif mode == 2:
            self.bicubic_interpolation_scale(scale_factor)

    def reset_to_original(self):
        """Сброс к оригинальному изображению"""
        if self.original_surface:
            self.image = self.original_surface
            self.scaled_surface = None


class BMViewer:
    """Основной класс приложения для просмотра BMP"""

    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("BMViewer - Оптимизированное сравнение методов интерполяции")
        self.clock = pygame.time.Clock()

        # Загрузчик BMP
        self.bmp_loader = BMPLoader()

        # Режимы масштабирования
        self.scaling_modes = [
            "Ближайший сосед",
            "Линейная интерполяция",
            "Бикубическая интерполяция"
        ]
        self.current_mode = 0

        # Коэффициент масштабирования
        self.scale_factor = 3.0
        self.min_scale = 0.5
        self.max_scale = 8.0

        # Тестовые изображения
        self.test_images = []
        self.current_test_image = 0
        self.load_test_images()

        # Кеширование результатов масштабирования
        self.scale_cache = {}
        self.last_scale_key = None

        # Флаги для оптимизации
        self.needs_redraw = True

    def load_test_images(self):
        """Загрузка тестовых изображений"""
        self.create_optimized_test_images()

    def create_optimized_test_images(self):
        """Создание оптимизированных тестовых изображений"""
        # Используем меньшие размеры для скорости, но сохраняем детализацию
        self.test_images = [
            ("Мелкая шахматка", self.create_fine_chessboard(48, 48)),
            ("Тонкие линии", self.create_thin_lines_image(80, 80)),
            ("Детальный градиент", self.create_detailed_gradient(96, 96)),
            ("Мелкий текст", self.create_text_image(120, 60))
        ]

    def create_fine_chessboard(self, width, height):
        """Создание мелкой шахматной доски (оптимизированная версия)"""
        surface = pygame.Surface((width, height))
        array = pygame.surfarray.pixels3d(surface)

        tile_size = 6
        # Векторизованное создание шахматной доски
        x_indices = np.arange(width)
        y_indices = np.arange(height)

        # Создаем шахматный паттерн используя broadcasting
        chess_mask = ((x_indices[:, None] // tile_size + y_indices[None, :] // tile_size) % 2 == 0)

        array[chess_mask] = [255, 255, 255]  # Белые клетки
        array[~chess_mask] = [0, 0, 0]  # Черные клетки

        del array
        return surface

    def create_thin_lines_image(self, width, height):
        """Создание изображения с тонкими линиями (оптимизированная версия)"""
        surface = pygame.Surface((width, height))
        surface.fill((255, 255, 255))

        # Рисуем линии сразу на поверхности
        for i in range(0, width, 5):
            pygame.draw.line(surface, (0, 0, 0), (i, 0), (i, height), 1)
        for i in range(0, height, 5):
            pygame.draw.line(surface, (0, 0, 0), (0, i), (width, i), 1)

        # Меньше диагональных линий для скорости
        for i in range(-height, width, 10):
            pygame.draw.line(surface, (255, 0, 0), (i, 0), (i + height, height), 1)

        return surface

    def create_detailed_gradient(self, width, height):
        """Создание градиента с мелкими деталями (оптимизированная версия)"""
        surface = pygame.Surface((width, height))
        array = pygame.surfarray.pixels3d(surface)

        # Векторизованный градиент
        x_indices = np.arange(width)
        y_indices = np.arange(height)

        array[:, :, 0] = 0  # Красный канал
        array[:, :, 1] = (y_indices[None, :] * 255 // (height - 1))  # Зеленый
        array[:, :, 2] = (x_indices[:, None] * 255 // (width - 1))  # Синий

        # Добавляем детали используя маски
        detail_mask = ((x_indices[:, None] % 4 == 0) & (y_indices[None, :] % 4 == 0))
        array[detail_mask] = [255, 255, 255]

        detail_mask2 = ((x_indices[:, None] % 4 == 2) & (y_indices[None, :] % 4 == 2))
        array[detail_mask2] = [0, 0, 0]

        del array
        return surface

    def create_text_image(self, width, height):
        """Создание изображения с мелким текстом (оптимизированная версия)"""
        surface = pygame.Surface((width, height))
        surface.fill((240, 240, 240))

        font = pygame.font.SysFont('Arial', 9, bold=True)  # Чуть крупнее для читаемости

        texts = [
            "ABCDEFGHIJKLM",
            "NOPQRSTUVWXYZ",
            "abcdefghijklm",
            "nopqrstuvwxyz"
        ]

        for i, text in enumerate(texts):
            text_surface = font.render(text, True, (0, 0, 0))
            surface.blit(text_surface, (5, 5 + i * 10))

        return surface

    def get_scale_key(self, scale_factor, mode, image_index):
        """Генерирует ключ для кеширования"""
        return f"{image_index}_{mode}_{scale_factor:.2f}"

    def load_current_test_image(self):
        """Загрузка текущего тестового изображения"""
        if self.test_images:
            name, surface = self.test_images[self.current_test_image]
            self.bmp_loader.original_surface = surface
            self.bmp_loader.image = surface
            self.bmp_loader.filename = f"{name}"
            self.bmp_loader.original_array = pygame.surfarray.array3d(surface)

            # Очищаем кеш при смене изображения
            self.scale_cache.clear()
            self.needs_redraw = True

    def apply_scaling_with_cache(self):
        """Применение масштабирования с кешированием"""
        scale_key = self.get_scale_key(self.scale_factor, self.current_mode, self.current_test_image)

        if scale_key in self.scale_cache:
            # Используем кешированный результат
            self.bmp_loader.image = self.scale_cache[scale_key]
        else:
            # Вычисляем и кешируем результат
            self.bmp_loader.apply_scaling(self.scale_factor, self.current_mode)
            self.scale_cache[scale_key] = self.bmp_loader.image

        self.last_scale_key = scale_key
        self.needs_redraw = True

    def draw_comparison(self):
        """Отрисовка сравнения всех трех методов (оптимизированная версия)"""
        if not self.bmp_loader.original_surface:
            return

        original_width = self.bmp_loader.original_surface.get_width()
        original_height = self.bmp_loader.original_surface.get_height()

        # Позиции для отрисовки
        start_x = 50
        start_y = 150
        spacing = 15  # Меньше расстояния для компактности

        # Предварительно вычисляем все масштабированные версии
        scaled_surfaces = []
        original_mode = self.current_mode

        for mode in range(3):
            scale_key = self.get_scale_key(self.scale_factor, mode, self.current_test_image)

            if scale_key in self.scale_cache:
                scaled_surfaces.append(self.scale_cache[scale_key])
            else:
                self.current_mode = mode
                self.bmp_loader.apply_scaling(self.scale_factor, mode)
                scaled_surface = self.bmp_loader.image
                self.scale_cache[scale_key] = scaled_surface
                scaled_surfaces.append(scaled_surface)

        # Восстанавливаем текущий режим
        self.current_mode = original_mode
        self.bmp_loader.image = scaled_surfaces[self.current_mode]

        # Отрисовываем все версии
        x_positions = [start_x, start_x + original_width + spacing]

        # Оригинал
        self.draw_image_with_label(self.bmp_loader.original_surface, "Оригинал",
                                   x_positions[0], start_y, original_width, original_height)

        # Все методы масштабирования
        labels = ["Ближайший сосед", "Линейная", "Бикубическая"]
        current_x = x_positions[1]

        for i, surface in enumerate(scaled_surfaces):
            self.draw_image_with_label(surface, labels[i],
                                       current_x, start_y, surface.get_width(), surface.get_height())
            current_x += surface.get_width() + spacing

    def draw_image_with_label(self, surface, label, x, y, width, height):
        """Отрисовка изображения с подписью"""
        # Подпись
        font = pygame.font.SysFont('Arial', 14)
        label_surface = font.render(label, True, TEXT_COLOR)
        self.screen.blit(label_surface, (x, y - 25))

        # Изображение
        self.screen.blit(surface, (x, y))

        # Размер
        size_surface = font.render(f"{width}x{height}", True, TEXT_COLOR)
        self.screen.blit(size_surface, (x, y + height + 5))

        # Рамка
        pygame.draw.rect(self.screen, HIGHLIGHT_COLOR,
                         (x - 2, y - 2, width + 4, height + 4), 1)

    def draw_interface(self):
        """Отрисовка интерфейса"""
        # Панель управления
        panel_rect = pygame.Rect(0, 0, SCREEN_WIDTH, 120)
        pygame.draw.rect(self.screen, PANEL_COLOR, panel_rect)

        # Заголовок
        font_large = pygame.font.SysFont('Arial', 24, bold=True)
        title = font_large.render("Оптимизированное сравнение методов интерполяции", True, TEXT_COLOR)
        self.screen.blit(title, (20, 20))

        # Информация о тестовом изображении
        font_small = pygame.font.SysFont('Arial', 16)
        if self.test_images:
            name, _ = self.test_images[self.current_test_image]
            image_info = font_small.render(f"Тест: {name}", True, HIGHLIGHT_COLOR)
            self.screen.blit(image_info, (20, 50))

        # Коэффициент масштабирования
        scale_info = font_small.render(f"Масштаб: {self.scale_factor:.1f}x", True, TEXT_COLOR)
        self.screen.blit(scale_info, (20, 70))

        # Размер кеша
        cache_info = font_small.render(f"Кеш: {len(self.scale_cache)} изображений", True, (150, 200, 150))
        self.screen.blit(cache_info, (20, 90))

        # Инструкция
        instructions = [
            "1-3: Выбор метода | Q/E: Масштаб | T: Следующий тест | R: Сброс | ESC: Выход"
        ]

        instruction_surface = font_small.render(instructions[0], True, (200, 200, 100))
        self.screen.blit(instruction_surface, (400, 90))

    def handle_events(self):
        """Обработка событий"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                elif event.key == K_1:
                    self.current_mode = 0
                    self.apply_scaling_with_cache()
                elif event.key == K_2:
                    self.current_mode = 1
                    self.apply_scaling_with_cache()
                elif event.key == K_3:
                    self.current_mode = 2
                    self.apply_scaling_with_cache()
                elif event.key == K_q:
                    self.scale_factor = max(self.min_scale, self.scale_factor - 0.5)
                    self.apply_scaling_with_cache()
                elif event.key == K_e:
                    self.scale_factor = min(self.max_scale, self.scale_factor + 0.5)
                    self.apply_scaling_with_cache()
                elif event.key == K_r:
                    self.scale_factor = 1.0
                    self.apply_scaling_with_cache()
                elif event.key == K_t:
                    self.current_test_image = (self.current_test_image + 1) % len(self.test_images)
                    self.load_current_test_image()

        return True

    def run(self):
        """Основной цикл приложения"""
        self.load_current_test_image()

        running = True
        while running:
            running = self.handle_events()

            if self.needs_redraw:
                self.screen.fill(BACKGROUND_COLOR)
                self.draw_interface()
                self.draw_comparison()
                pygame.display.flip()
                self.needs_redraw = False

            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    app = BMViewer()
    app.run()