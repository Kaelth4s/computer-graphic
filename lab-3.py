import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
import matplotlib.patches as patches
from typing import List, Tuple, Union, Optional
import math


class BezierCurve:
    """
    Класс для работы с кривыми Безье
    """

    def __init__(self, control_points: List[Tuple[float, float]]):
        """
        Инициализация кривой Безье

        Args:
            control_points: Список контрольных точек [(x1, y1), (x2, y2), ...]
        """
        self.control_points = np.array(control_points)
        self.n = len(control_points) - 1  # Степень кривой

    def binomial_coefficient(self, n: int, k: int) -> int:
        """Вычисление биномиального коэффициента C(n, k)"""
        return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

    def bernstein_polynomial(self, n: int, k: int, t: float) -> float:
        """Вычисление полинома Бернштейна B_{n,k}(t)"""
        return self.binomial_coefficient(n, k) * (t ** k) * ((1 - t) ** (n - k))

    def evaluate(self, t: float) -> Tuple[float, float]:
        """
        Вычисление точки на кривой Безье для параметра t ∈ [0, 1]

        Args:
            t: Параметр вдоль кривой (0 - начало, 1 - конец)

        Returns:
            Координаты точки (x, y)
        """
        if not (0 <= t <= 1):
            raise ValueError("Параметр t должен быть в диапазоне [0, 1]")

        point = np.zeros(2)
        for k in range(self.n + 1):
            bernstein = self.bernstein_polynomial(self.n, k, t)
            point += bernstein * self.control_points[k]

        return tuple(point)

    def evaluate_multiple(self, t_values: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Вычисление нескольких точек на кривой

        Args:
            t_values: Массив параметров t

        Returns:
            Массив точек на кривой
        """
        t_array = np.array(t_values)
        points = np.zeros((len(t_array), 2))

        for i, t in enumerate(t_array):
            points[i] = self.evaluate(t)

        return points

    def get_curve_points(self, num_points: int = 100) -> np.ndarray:
        """
        Получение равномерно распределенных точек вдоль кривой

        Args:
            num_points: Количество точек для генерации

        Returns:
            Массив точек кривой
        """
        t_values = np.linspace(0, 1, num_points)
        return self.evaluate_multiple(t_values)

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Получение ограничивающего прямоугольника"""
        points = self.control_points
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        return x_min, x_max, y_min, y_max

    def translate(self, dx: float, dy: float):
        """Перемещение кривой"""
        self.control_points[:, 0] += dx
        self.control_points[:, 1] += dy

    def scale(self, sx: float, sy: float, center: Optional[Tuple[float, float]] = None):
        """Масштабирование кривой"""
        if center is None:
            center = np.mean(self.control_points, axis=0)

        cx, cy = center
        self.control_points[:, 0] = cx + (self.control_points[:, 0] - cx) * sx
        self.control_points[:, 1] = cy + (self.control_points[:, 1] - cy) * sy


class QuadraticBezier(BezierCurve):
    """Квадратичная кривая Безье (степень 2)"""

    def __init__(self, p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]):
        super().__init__([p0, p1, p2])

    def evaluate(self, t: float) -> Tuple[float, float]:
        """Оптимизированное вычисление для квадратичной кривой"""
        p0, p1, p2 = self.control_points
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        return (x, y)


class CubicBezier(BezierCurve):
    """Кубическая кривая Безье (степень 3)"""

    def __init__(self, p0: Tuple[float, float], p1: Tuple[float, float],
                 p2: Tuple[float, float], p3: Tuple[float, float]):
        super().__init__([p0, p1, p2, p3])

    def evaluate(self, t: float) -> Tuple[float, float]:
        """Оптимизированное вычисление для кубической кривой"""
        p0, p1, p2, p3 = self.control_points
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        x = mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0]
        y = mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1]
        return (x, y)


def de_casteljau(control_points: List[Tuple[float, float]], t: float) -> Tuple[float, float]:
    """
    Рекурсивная реализация алгоритма де Кастельжо

    Args:
        control_points: Контрольные точки
        t: Параметр вдоль кривой

    Returns:
        Точка на кривой Безье
    """
    points = np.array(control_points)

    def recursive_de_casteljau(points, t):
        if len(points) == 1:
            return points[0]

        new_points = []
        for i in range(len(points) - 1):
            x = (1 - t) * points[i][0] + t * points[i + 1][0]
            y = (1 - t) * points[i][1] + t * points[i + 1][1]
            new_points.append((x, y))

        return recursive_de_casteljau(new_points, t)

    return recursive_de_casteljau(points, t)


def de_casteljau_iterative(control_points: List[Tuple[float, float]], t: float) -> Tuple[float, float]:
    """
    Итеративная реализация алгоритма де Кастельжо
    """
    points = np.array(control_points).copy()
    n = len(points)

    for r in range(1, n):
        for i in range(n - r):
            points[i] = (1 - t) * points[i] + t * points[i + 1]

    return tuple(points[0])


def plot_bezier_curve(bezier_curve: BezierCurve, num_points: int = 100,
                      show_control_points: bool = True,
                      show_control_polygon: bool = True):
    """
    Визуализация кривой Безье

    Args:
        bezier_curve: Объект кривой Безье
        num_points: Количество точек для отрисовки
        show_control_points: Показывать контрольные точки
        show_control_polygon: Показывать контрольный полигон
    """
    plt.figure(figsize=(10, 8))

    # Получаем точки кривой
    curve_points = bezier_curve.get_curve_points(num_points)

    # Рисуем кривую
    plt.plot(curve_points[:, 0], curve_points[:, 1], 'b-', linewidth=2, label='Кривая Безье')

    # Рисуем контрольные точки и полигон
    if show_control_points:
        control_points = bezier_curve.control_points
        plt.plot(control_points[:, 0], control_points[:, 1], 'ro--',
                 linewidth=1, markersize=8, label='Контрольные точки')

        for i, point in enumerate(control_points):
            plt.annotate(f'P{i}', (point[0], point[1]),
                         xytext=(5, 5), textcoords='offset points', fontsize=12)

    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.title(f'Кривая Безье степени {bezier_curve.n}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def plot_multiple_curves(curves: List[BezierCurve], colors: List[str] = None):
    """Отрисовка нескольких кривых Безье"""
    if colors is None:
        colors = ['blue', 'red', 'green', 'orange', 'purple']

    plt.figure(figsize=(12, 8))

    for i, curve in enumerate(curves):
        color = colors[i % len(colors)]
        curve_points = curve.get_curve_points()

        # Кривая
        plt.plot(curve_points[:, 0], curve_points[:, 1],
                 color=color, linewidth=2, label=f'Кривая {i + 1}')

        # Контрольные точки
        control_points = curve.control_points
        plt.plot(control_points[:, 0], control_points[:, 1],
                 color=color, marker='o', linestyle='--', alpha=0.5)

    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.legend()
    plt.title('Несколько кривых Безье')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def bezier_derivative(control_points: List[Tuple[float, float]], t: float) -> Tuple[float, float]:
    """
    Вычисление производной (касательного вектора) кривой Безье
    """
    n = len(control_points) - 1
    points = np.array(control_points)

    derivative = np.zeros(2)
    for k in range(n):
        bernstein = BezierCurve([]).bernstein_polynomial(n - 1, k, t)
        diff = n * (points[k + 1] - points[k])
        derivative += bernstein * diff

    return tuple(derivative)


def split_bezier_curve(control_points: List[Tuple[float, float]], t: float):
    """
    Разделение кривой Безье на две части в точке t
    Возвращает контрольные точки для двух новых кривых
    """
    points = np.array(control_points)
    n = len(points)

    # Используем алгоритм де Кастельжо для разделения
    left_points = []
    right_points = []

    current_points = points.copy()
    left_points.append(tuple(current_points[0]))
    right_points.append(tuple(current_points[-1]))

    for r in range(1, n):
        new_points = []
        for i in range(n - r):
            current_points[i] = (1 - t) * current_points[i] + t * current_points[i + 1]
            new_points.append(tuple(current_points[i]))

        left_points.append(tuple(current_points[0]))
        right_points.insert(0, tuple(current_points[n - r - 1]))

    return left_points, right_points


class InteractiveBezierEditor:
    """
    Интерактивный редактор кривых Безье
    """

    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(bottom=0.3)

        self.current_curve = None
        self.curves = []  # Все кривые на холсте
        self.selected_curve_idx = -1
        self.selected_point_idx = -1
        self.dragging = False

        # Начальные контрольные точки
        self.control_points = [
            [(0.1, 0.1), (0.3, 0.7), (0.7, 0.8), (0.9, 0.2)],  # Кубическая кривая
            [(0.1, 0.5), (0.5, 0.9), (0.9, 0.5)]  # Квадратичная кривая
        ]

        self.setup_ui()
        self.update_plot()

    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Кнопки управления кривыми
        self.ax_add_curve = plt.axes([0.05, 0.15, 0.1, 0.04])
        self.btn_add_curve = Button(self.ax_add_curve, 'Добавить кривую')
        self.btn_add_curve.on_clicked(self.add_curve)

        self.ax_remove_curve = plt.axes([0.16, 0.15, 0.1, 0.04])
        self.btn_remove_curve = Button(self.ax_remove_curve, 'Удалить кривую')
        self.btn_remove_curve.on_clicked(self.remove_curve)

        self.ax_clear_all = plt.axes([0.27, 0.15, 0.1, 0.04])
        self.btn_clear_all = Button(self.ax_clear_all, 'Очистить всё')
        self.btn_clear_all.on_clicked(self.clear_all)

        # Кнопки трансформаций
        self.ax_translate_x = plt.axes([0.05, 0.1, 0.08, 0.04])
        self.btn_translate_x = Button(self.ax_translate_x, 'Сдвиг X+0.1')
        self.btn_translate_x.on_clicked(lambda x: self.translate_current(0.1, 0))

        self.ax_translate_y = plt.axes([0.14, 0.1, 0.08, 0.04])
        self.btn_translate_y = Button(self.ax_translate_y, 'Сдвиг Y+0.1')
        self.btn_translate_y.on_clicked(lambda x: self.translate_current(0, 0.1))

        self.ax_scale_up = plt.axes([0.23, 0.1, 0.08, 0.04])
        self.btn_scale_up = Button(self.ax_scale_up, 'Масштаб +')
        self.btn_scale_up.on_clicked(lambda x: self.scale_current(1.2))

        self.ax_scale_down = plt.axes([0.32, 0.1, 0.08, 0.04])
        self.btn_scale_down = Button(self.ax_scale_down, 'Масштаб -')
        self.btn_scale_down.on_clicked(lambda x: self.scale_current(0.8))

        # Слайдер для степени кривой
        self.ax_degree = plt.axes([0.45, 0.15, 0.2, 0.03])
        self.slider_degree = Slider(
            self.ax_degree, 'Степень кривой', 1, 7, valinit=3, valstep=1
        )
        self.slider_degree.on_changed(self.change_degree)

        # Поле для количества точек
        self.ax_num_points = plt.axes([0.45, 0.1, 0.15, 0.03])
        self.text_num_points = TextBox(self.ax_num_points, 'Точек:', initial='100')
        self.text_num_points.on_submit(self.change_num_points)

        # Кнопки экспорта и информации
        self.ax_export = plt.axes([0.65, 0.15, 0.1, 0.04])
        self.btn_export = Button(self.ax_export, 'Экспорт данных')
        self.btn_export.on_clicked(self.export_data)

        self.ax_info = plt.axes([0.76, 0.15, 0.1, 0.04])
        self.btn_info = Button(self.ax_info, 'Справка')
        self.btn_info.on_clicked(self.show_help)

        # Настройка событий мыши
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Настройка осей
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Интерактивный редактор кривых Безье\n'
                          '• Клик по точке: выбрать/перемещать\n'
                          '• Двойной клик: добавить точку\n'
                          '• Правый клик по точке: удалить\n'
                          '• Delete: удалить выбранную точку')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        # Информационная панель
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def add_curve(self, event=None):
        """Добавление новой кривой"""
        # Создаем простую кривую по умолчанию
        base_x = 0.1 + len(self.control_points) * 0.15
        base_y = 0.2 + (len(self.control_points) % 3) * 0.1
        new_points = [
            (base_x, base_y),
            (base_x + 0.2, base_y + 0.6),
            (base_x + 0.4, base_y),
            (base_x + 0.6, base_y + 0.3)
        ]

        self.control_points.append(new_points[:3])  # Начинаем с квадратичной
        self.selected_curve_idx = len(self.control_points) - 1
        self.slider_degree.set_val(2)  # Устанавливаем степень 2
        self.update_plot()

    def remove_curve(self, event=None):
        """Удаление текущей кривой"""
        if self.control_points and 0 <= self.selected_curve_idx < len(self.control_points):
            self.control_points.pop(self.selected_curve_idx)
            self.selected_curve_idx = min(self.selected_curve_idx, len(self.control_points) - 1)
            self.update_plot()

    def clear_all(self, event=None):
        """Очистка всех кривых"""
        self.control_points.clear()
        self.selected_curve_idx = -1
        self.update_plot()

    def translate_current(self, dx, dy):
        """Сдвиг выбранной кривой"""
        if 0 <= self.selected_curve_idx < len(self.control_points):
            points = self.control_points[self.selected_curve_idx]
            for i in range(len(points)):
                x, y = points[i]
                points[i] = (x + dx, y + dy)
            self.update_plot()

    def scale_current(self, factor):
        """Масштабирование выбранной кривой"""
        if 0 <= self.selected_curve_idx < len(self.control_points):
            points = np.array(self.control_points[self.selected_curve_idx])
            center = np.mean(points, axis=0)

            scaled_points = []
            for point in points:
                dx = point[0] - center[0]
                dy = point[1] - center[1]
                scaled_points.append((center[0] + dx * factor, center[1] + dy * factor))

            self.control_points[self.selected_curve_idx] = scaled_points
            self.update_plot()

    def change_degree(self, val):
        """Изменение степени выбранной кривой"""
        if 0 <= self.selected_curve_idx < len(self.control_points):
            current_points = self.control_points[self.selected_curve_idx]
            new_degree = int(val)

            if new_degree + 1 > len(current_points):
                # Добавляем точки
                while len(current_points) < new_degree + 1:
                    last_point = current_points[-1]
                    current_points.append((last_point[0] + 0.05, last_point[1] + 0.05))
            elif new_degree + 1 < len(current_points):
                # Удаляем точки
                self.control_points[self.selected_curve_idx] = current_points[:new_degree + 1]

            self.update_plot()

    def change_num_points(self, text):
        """Изменение количества точек для отрисовки"""
        self.update_plot()

    def export_data(self, event):
        """Экспорт данных кривых"""
        print("\n" + "=" * 50)
        print("ЭКСПОРТ ДАННЫХ КРИВЫХ БЕЗЬЕ")
        print("=" * 50)

        if not self.control_points:
            print("Нет кривых для экспорта")
            return

        for i, points in enumerate(self.control_points):
            print(f"\n--- Кривая {i + 1} (степень {len(points) - 1}) ---")
            for j, (x, y) in enumerate(points):
                print(f"P{j}: ({x:.4f}, {y:.4f})")

            # Создаем объект кривой для вычисления длины
            curve = BezierCurve(points)
            curve_points = curve.get_curve_points(100)
            length = np.sum(np.sqrt(np.sum(np.diff(curve_points, axis=0) ** 2, axis=1)))
            print(f"Длина кривой: {length:.4f}")

        print("\nКод для воссоздания кривых:")
        print("control_points = [")
        for points in self.control_points:
            print("    " + str([(round(x, 3), round(y, 3)) for x, y in points]) + ",")
        print("]")

    def show_help(self, event):
        """Показать справку"""
        help_text = """
        ИНТЕРАКТИВНЫЙ РЕДАКТОР КРИВЫХ БЕЗЬЕ

        Управление мышью:
        • Левый клик по точке - выбрать/перемещать
        • Двойной клик на пустом месте - добавить точку к выбранной кривой
        • Правый клик по точке - удалить точку
        • Перетаскивание точки - изменение формы кривой

        Горячие клавиши:
        • Delete - удалить выбранную точку
        • +/- - масштабировать кривую
        • Стрелки - переместить кривую

        Кнопки управления:
        • Добавить/Удалить кривую - управление кривыми
        • Сдвиг X/Y - перемещение кривой
        • Масштаб +/- - изменение размера
        • Степень кривой - изменение количества контрольных точек
        """
        print(help_text)

    def find_nearest_point(self, x, y, threshold=0.05):
        """Поиск ближайшей контрольной точки"""
        for curve_idx, points in enumerate(self.control_points):
            for point_idx, (px, py) in enumerate(points):
                distance = np.sqrt((px - x) ** 2 + (py - y) ** 2)
                if distance < threshold:
                    return curve_idx, point_idx
        return -1, -1

    def on_click(self, event):
        """Обработка клика мыши"""
        if event.inaxes != self.ax:
            return

        if event.dblclick:
            # Двойной клик - добавление точки
            if 0 <= self.selected_curve_idx < len(self.control_points):
                self.control_points[self.selected_curve_idx].append((event.xdata, event.ydata))
                # Обновляем слайдер степени
                new_degree = len(self.control_points[self.selected_curve_idx]) - 1
                self.slider_degree.set_val(new_degree)
                self.update_plot()
            return

        if event.button == 1:  # Левый клик
            curve_idx, point_idx = self.find_nearest_point(event.xdata, event.ydata)
            if curve_idx != -1:
                self.selected_curve_idx = curve_idx
                self.selected_point_idx = point_idx
                self.dragging = True
                self.update_plot()
            else:
                self.selected_point_idx = -1
                self.update_plot()

        elif event.button == 3:  # Правый клик - удаление точки
            curve_idx, point_idx = self.find_nearest_point(event.xdata, event.ydata)
            if (curve_idx != -1 and point_idx != -1 and
                    len(self.control_points[curve_idx]) > 2):
                self.control_points[curve_idx].pop(point_idx)
                self.selected_point_idx = -1
                # Обновляем слайдер степени
                if curve_idx == self.selected_curve_idx:
                    new_degree = len(self.control_points[curve_idx]) - 1
                    self.slider_degree.set_val(new_degree)
                self.update_plot()

    def on_release(self, event):
        """Обработка отпускания кнопки мыши"""
        self.dragging = False

    def on_motion(self, event):
        """Обработка движения мыши"""
        if (self.dragging and event.inaxes == self.ax and
                self.selected_curve_idx != -1 and self.selected_point_idx != -1):
            # Обновляем позицию точки
            self.control_points[self.selected_curve_idx][self.selected_point_idx] = (
                event.xdata, event.ydata
            )
            self.update_plot()

    def on_key_press(self, event):
        """Обработка нажатия клавиш"""
        if event.key == 'delete':
            # Удаление выбранной точки
            if (self.selected_curve_idx != -1 and self.selected_point_idx != -1 and
                    len(self.control_points[self.selected_curve_idx]) > 2):
                self.control_points[self.selected_curve_idx].pop(self.selected_point_idx)
                self.selected_point_idx = -1
                self.update_plot()

        elif event.key == '+':
            self.scale_current(1.1)
        elif event.key == '-':
            self.scale_current(0.9)
        elif event.key == 'right':
            self.translate_current(0.05, 0)
        elif event.key == 'left':
            self.translate_current(-0.05, 0)
        elif event.key == 'up':
            self.translate_current(0, 0.05)
        elif event.key == 'down':
            self.translate_current(0, -0.05)

    def update_plot(self):
        """Обновление графика"""
        self.ax.clear()
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        self.ax.set_title('Интерактивный редактор кривых Безье')

        num_points = int(self.text_num_points.text) if self.text_num_points.text.isdigit() else 100

        # Отрисовываем все кривые
        total_length = 0
        for curve_idx, points in enumerate(self.control_points):
            if len(points) < 2:
                continue

            # Создаем кривую Безье
            curve = BezierCurve(points)
            curve_points = curve.get_curve_points(num_points)

            # Вычисляем длину кривой
            length = np.sum(np.sqrt(np.sum(np.diff(curve_points, axis=0) ** 2, axis=1)))
            total_length += length

            # Определяем цвет в зависимости от выбора
            color = 'red' if curve_idx == self.selected_curve_idx else 'blue'
            linewidth = 3 if curve_idx == self.selected_curve_idx else 2

            # Рисуем кривую
            self.ax.plot(curve_points[:, 0], curve_points[:, 1],
                         color=color, linewidth=linewidth,
                         label=f'Кривая {curve_idx + 1} (ст.{len(points) - 1})')

            # Рисуем контрольные точки и полигон
            points_array = np.array(points)
            self.ax.plot(points_array[:, 0], points_array[:, 1],
                         'o--', color=color, alpha=0.7, markersize=8)

            # Подписываем точки
            for point_idx, (x, y) in enumerate(points):
                point_color = 'green' if (curve_idx == self.selected_curve_idx and
                                          point_idx == self.selected_point_idx) else color
                self.ax.plot(x, y, 'o', color=point_color, markersize=10)
                self.ax.annotate(f'P{point_idx}', (x, y),
                                 xytext=(8, 8), textcoords='offset points',
                                 fontsize=9, color=point_color, weight='bold')

        # Обновляем информационную панель
        info_text = f'Кривые: {len(self.control_points)} | '
        info_text += f'Общая длина: {total_length:.3f}\n'

        if 0 <= self.selected_curve_idx < len(self.control_points):
            points = self.control_points[self.selected_curve_idx]
            info_text += f'Выбрана кривая {self.selected_curve_idx + 1}: '
            info_text += f'степень {len(points) - 1}, точек: {len(points)}'

            if self.selected_point_idx != -1:
                x, y = points[self.selected_point_idx]
                info_text += f'\nТочка P{self.selected_point_idx}: ({x:.3f}, {y:.3f})'

        self.info_text.set_text(info_text)

        self.ax.legend(loc='upper right')
        self.fig.canvas.draw_idle()


def demo_bezier_curves():
    """Демонстрация работы кривых Безье"""
    print("ДЕМОНСТРАЦИЯ КРИВЫХ БЕЗЬЕ")
    print("=" * 50)

    # Пример 1: Квадратичная кривая Безье
    print("\n1. Квадратичная кривая Безье:")
    quadratic = QuadraticBezier((0, 0), (1, 2), (3, 1))

    # Вычисляем несколько точек
    for t in [0, 0.25, 0.5, 0.75, 1]:
        point = quadratic.evaluate(t)
        point_de_casteljau = de_casteljau(quadratic.control_points.tolist(), t)
        print(f"t={t}: Бернштейн: ({point[0]:.3f}, {point[1]:.3f}) | "
              f"Де Кастельжо: ({point_de_casteljau[0]:.3f}, {point_de_casteljau[1]:.3f})")

    # Пример 2: Кубическая кривая Безье
    print("\n2. Кубическая кривая Безье:")
    cubic = CubicBezier((0, 0), (1, 3), (4, 3), (5, 0))

    # Сравнение методов
    t = 0.7
    point1 = cubic.evaluate(t)
    point2 = de_casteljau(cubic.control_points.tolist(), t)
    print(f"t=0.7: Метод Бернштейна: ({point1[0]:.3f}, {point1[1]:.3f})")
    print(f"       Де Кастельжо:    ({point2[0]:.3f}, {point2[1]:.3f})")

    # Визуализация
    plot_bezier_curve(quadratic, show_control_points=True)
    plot_bezier_curve(cubic, show_control_points=True)

    # Пример 3: Несколько кривых
    print("\n3. Несколько кривых Безье:")
    curve1 = QuadraticBezier((0, 0), (1, 2), (2, 0))
    curve2 = CubicBezier((2, 0), (3, 2), (4, -1), (5, 1))
    curve3 = BezierCurve([(5, 1), (6, 3), (7, 0), (8, 2)])

    plot_multiple_curves([curve1, curve2, curve3])


def run_interactive_editor():
    """Запуск интерактивного редактора"""
    print("\n" + "=" * 60)
    print("ЗАПУСК ИНТЕРАКТИВНОГО РЕДАКТОРА КРИВЫХ БЕЗЬЕ")
    print("=" * 60)
    print("\nИнструкции:")
    print("• Левый клик: выбрать/перемещать точки")
    print("• Двойной клик: добавить точку к выбранной кривой")
    print("• Правый клик по точке: удалить точку")
    print("• Delete: удалить выбранную точку")
    print("• Стрелки: переместить кривую")
    print("• +/-: масштабировать кривую")
    print("\nИспользуйте кнопки для дополнительного управления!")

    editor = InteractiveBezierEditor()
    plt.show()
    return editor


def main():
    """Главная функция"""
    print("ПРОГРАММА ДЛЯ РАБОТЫ С КРИВЫМИ БЕЗЬЕ")
    print("=" * 50)

    while True:
        print("\nВыберите режим:")
        print("1 - Демонстрация кривых Безье")
        print("2 - Интерактивный редактор")
        print("3 - Выход")

        choice = input("Ваш выбор (1-3): ").strip()

        if choice == '1':
            demo_bezier_curves()
        elif choice == '2':
            run_interactive_editor()
        elif choice == '3':
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")


if __name__ == "__main__":
    main()