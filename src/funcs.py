import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict


def extract_and_check(value, thresh: float) -> bool:
    """
    Проверяет, если p-value ниже заданного порога.

    :param value: Строка, содержащая p-value в скобках.
    :param thresh: Пороговое значение для проверки.
    :return: True, если p-value меньше порога, иначе False.
    """
    match = re.search(r'\((.*?)\)', value)
    if match:
        num = float(match.group(1))
        if num < thresh:
            return True
    return False


def get_chans_connection(df: pd.DataFrame,
                         thresh: float,
                         df_path_or_name='',
                         connection_column_name='chan_connections') -> dict:
    """
    Возвращает словарь, где:
    - ключ - пара каналов,
    - значение - 'сила связи'.

    :param df: DataFrame с данными о каналах.
    :param thresh: Пороговое значение для фильтрации.
    :param df_path_or_name: Путь к файлу или имя DataFrame для отладки.
    :param connection_column_name: Название столбца с информацией о соединениях каналов.
    :return: Словарь пар каналов и силы их связи.
    """
    # Проверка наличия столбца с соединениями в DataFrame
    assert connection_column_name in df.columns, \
        f"DataFrame должен содержать столбец {connection_column_name}."

    # Проверка соответствия значений паттерну
    for val in df[connection_column_name]:
        assert re.match(r'^[a-zA-Z]\d+[‐-][a-zA-Z]\d+$', val), \
            f"Неверное значение - {val} - в столбце {connection_column_name} " \
            f"DataFrame {df_path_or_name}."

    col_ls = df.columns[1:]
    dictionary = {}

    for col in col_ls:
        # Проверка соответствия значений в столбце паттерну
        for val in df[col]:
            try:
                assert re.match(r'^-?\d+(\.\d+)?\(\d+(\.\d+)?\)$', val)
            except AssertionError:
                raise ValueError(f"Неверное значение - {val} - в столбце '{col}' DataFrame {df_path_or_name}."
                                f"Ожидался формат значения - например: 1.1'(0.1)'")
        tmp_d = {}
        filtered_rows = df[df[col].apply(extract_and_check, thresh=thresh)]

        for _, row in filtered_rows.iterrows():
            cell_value = row[col]
            pair = tuple([row[connection_column_name],
                          float(cell_value.split('(')[0])])
            tmp_d[pair[0]] = pair[1]

        dictionary[col] = tmp_d

    return dictionary

def get_chans_connectionv2(df: pd.DataFrame,
                         thresh: float,
                         df_path_or_name='',
                         connection_column_name='chan_connections',
                         which_columns=None) -> dict:
    """
    Возвращает словарь, где:
    - ключ - пара каналов,
    - значение - 'сила связи'.

    Похож на функцию get_chans_connection, но позволяет указать конкретные столбцы для анализа.

    :param df: DataFrame с данными о каналах.
    :param thresh: Пороговое значение для фильтрации.
    :param df_path_or_name: Путь к файлу или имя DataFrame для отладки.
    :param connection_column_name: Название столбца с информацией о соединениях каналов.
    :param which_columns: Список столбцов для анализа (если None, анализируются все столбцы кроме первого).
    :return: Словарь пар каналов и силы их связи.
    """
    # Проверка наличия столбца с соединениями в DataFrame
    assert connection_column_name in df.columns, \
        f"DataFrame должен содержать столбец {connection_column_name}."

    # Проверка соответствия значений паттерну
    for val in df[connection_column_name]:
        assert re.match(r'^[a-zA-Z]\d+[‐-][a-zA-Z]\d+$', val), \
            f"Неверное значение - {val} - в столбце {connection_column_name} " \
            f"DataFrame {df_path_or_name}."

    if which_columns:
        col_ls = df.columns[which_columns]
    else: 
        col_ls = df.columns[1:]
    dictionary = {}

    for col in col_ls:
        # Проверка соответствия значений в столбце паттерну
        for val in df[col]:
            try:
                assert re.match(r'^-?\d+(\.\d+)?\(\d+(\.\d+)?\)$', val)
            except AssertionError:
                raise ValueError(f"Неверное значение - {val} - в столбце '{col}' DataFrame {df_path_or_name}."
                                f"Ожидался формат значения - например: 1.1'(0.1)'")
        tmp_d = {}
        filtered_rows = df[df[col].apply(extract_and_check, thresh=thresh)]

        for _, row in filtered_rows.iterrows():
            cell_value = row[col]
            pair = tuple([row[connection_column_name],
                          float(cell_value.split('(')[0])])
            tmp_d[pair[0]] = pair[1]

        dictionary[col] = tmp_d

    return dictionary

def count_connections(connections_dict, calculate_stats=False):
    """
    Подсчитывает количество соединений и, при необходимости, вычисляет статистику для каждого узла.

    :param connections_dict: Словарь соединений каналов, где ключ - строка с парой каналов,
                             значение - сила связи.
    :param calculate_stats: Если True, возвращает дополнительный словарь со статистикой по узлам.
    :return: Если calculate_stats=False, возвращает словарь с количеством соединений для каждого узла.
            Если calculate_stats=True, возвращает два словаря: один с количеством соединений,
            другой с вычисленной статистикой (веса узлов).
    """
    node_connections = defaultdict(int)
    node_values = defaultdict(list)

    for connection, value in connections_dict.items():
        nodes = connection.split('-')
        for node in nodes:
            node_connections[node] += 1
            node_values[node].append(value)

    if calculate_stats:
        node_stats = {}
        for node, values in node_values.items():
            weight = round((sum(values) / 3), 4)
            node_stats[node] = weight
    
        return dict(node_connections), node_stats
    else:
        return dict(node_connections)


def draw_lines(chan_connections: dict,
               chan_coords: dict,
               img_path: str,
               frequency='Not given',
               **kwargs) -> None:
    """
    Рисует линии между наиболее связанными каналами на изображении.

    Чем шире и менее прозрачна линия, тем сильнее связь между каналами.

    :param chan_connections: Словарь пар каналов и силы связи.
    :param chan_coords: Словарь координат каналов.
    :param img_path: Путь к изображению.
    :param frequency: Информация о частоте (по умолчанию 'Не указана').
    :param kwargs: Дополнительные параметры настройки:
        - line_color (str): Цвет линии (по умолчанию 'red').
        - base_alpha (float): Базовый альфа-канал для линии (по умолчанию 0.2).
        - base_lwidth (float): Базовая ширина линии (по умолчанию 0.3).
        - linestyle (str): Стиль линии (по умолчанию '-').
    """
    base_alpha = kwargs.get('base_alpha', 0.2)
    base_lwidth = kwargs.get('base_lwidth', 0.3)
    base_linestyle = kwargs.get('base_linestyle', '-')
    base_bicolormap = kwargs.get('base_bicolormap', plt.cm.bwr)
    base_pos_colormap = kwargs.get('base_pos_colormap', plt.cm.Reds)
    base_neg_colormap = kwargs.get('base_neg_colormap', plt.cm.Blues)

    # Определение диапазона значений в chan_connections
    values = np.array(list(chan_connections.values()))
    min_val = np.min(values)
    max_val = np.max(values)
    norm = plt.Normalize(vmin=min_val, vmax=max_val)

    # Отображение изображения
    image = mpimg.imread(img_path)
    plt.imshow(image)

    for key, val in chan_connections.items():
        k = tuple(key.split('-'))

        chan1 = chan_coords[k[0]]
        chan2 = chan_coords[k[1]]

        x_values = [chan1[0], chan2[0]]
        y_values = [chan1[1], chan2[1]]

        # Корректировка цвета, альфа-канала и ширины линии в зависимости от значений
        if min_val < 0 and max_val > 0:
            color_map = base_bicolormap  
            color_val = norm(val)
            line_color = color_map(color_val)
            alpha = base_alpha * abs(val) if base_alpha * abs(val) < 1 else 1
            linewidth = base_lwidth * abs(val)

        elif min_val >= 0:
            color_map = base_pos_colormap  
            color_val = norm(val)
            line_color = color_map(color_val)
            alpha = base_alpha * val if base_alpha * val < 1 else 1
            linewidth = base_lwidth * val

        else:
            color_map = base_neg_colormap
            color_val = norm(val)
            line_color = color_map(color_val)
            alpha = base_alpha * abs(val) if base_alpha * abs(val) < 1 else 1
            linewidth = base_lwidth * abs(val)

        plt.plot(x_values, y_values,
                 linestyle=base_linestyle,
                 color=line_color,
                 linewidth=linewidth,
                 alpha=alpha)

    plt.title('Частота: ' + frequency)


def draw_node_degree(node_degree: dict,
                     chans_coords: dict,
                     img_path: str,
                     frequency='Not given',
                     node_means_medians=None,
                     **kwargs) -> None:
    """
    Рисует узлы с изменяющимися размерами или цветами в зависимости от значений степени на изображении.

    :param node_degree: Словарь, отображающий узлы на их степени.
    :param chans_coords: Словарь, отображающий узлы на их координаты.
    :param img_path: Путь к изображению.
    :param frequency: Информация о частоте (по умолчанию 'Не указана').
    :param node_means_medians: Дополнительная информация о статистике узлов (необязательно).
    :param kwargs: Дополнительные параметры настройки:
        - base_alpha (float): Базовый альфа-канал для узлов (по умолчанию 0.2).
        - base_radius (float): Базовый радиус узлов (по умолчанию 6).
        - base_colormap (Colormap): Колormap для узлов (по умолчанию plt.cm.Reds).
        - fill (bool): Заполнить ли узлы (по умолчанию True).
    """
    base_alpha = kwargs.get('base_alpha', 0.2)
    base_circle_radius = kwargs.get('base_radius', 6)
    base_colormap = kwargs.get('base_colormap', plt.cm.Reds)
    fill_or_not = kwargs.get('fill', True)

    # Определение диапазона значений в node_degree
    values = np.array(list(node_degree.values()))
    min_val = np.min(values)
    max_val = np.max(values)
    norm = plt.Normalize(vmin=min_val, vmax=max_val)

    # Отображение изображения 
    image = mpimg.imread(img_path)
    plt.imshow(image)

    for key, val in node_degree.items():
        x, y = chans_coords[key]

        if node_means_medians is None:
            color = base_colormap(norm(val))
            alpha = base_alpha * abs(val) if base_alpha * abs(val) < 1 else 1
            radius = base_circle_radius * abs(val)  
        else:
            color = base_colormap(norm(val))
            alpha = base_alpha * abs(val) if base_alpha * abs(val) < 1 else 1
            radius = base_circle_radius * abs(val)  

        circle = plt.Circle((x, y),
                            radius=radius,
                            color=color,
                            fill=fill_or_not,
                            alpha=alpha)

        plt.gca().add_patch(circle)
    
    plt.title('Частота: ' + frequency)


def draw_network(chan_connections: dict,
                 node_degree: dict,
                 chans_coords: dict,
                 img_path: str,
                 frequency='Not given',
                 ax=None,
                 **kwargs) -> None:
    """
    Рисует линии между наиболее связанными каналами и узлы 
    с различными размерами или цветами в зависимости от значений степени.

    Параметры:
    - chan_connections (dict): Словарь, где ключи - это пары каналов (в формате 'канал1-канал2'),
      а значения - сила связи между ними.
    - node_degree (dict): Словарь, где ключи - это узлы, а значения - их степени.
    - chans_coords (dict): Словарь, где ключи - это каналы или узлы, а значения - их координаты в формате (x, y).
    - img_path (str): Путь к изображению, на которое будут наноситься линии и узлы.
    - frequency (str): Информация о частоте (по умолчанию 'Не указана').
    - ax (Axes, optional): Объект осей Matplotlib для рисования сети (по умолчанию None).
    - **kwargs: Дополнительные параметры настройки:
        - base_fontisize (int): Размер шрифта для заголовка (по умолчанию 10).
        - base_alpha (float): Базовый альфа-канал для линий и узлов (по умолчанию 0.2).
        - base_lwidth (float): Базовая ширина линии (по умолчанию 0.3).
        - base_linestyle (str): Стиль линии (по умолчанию '-').
        - base_bicolormap (Colormap): Колоркарта для двухцветных линий (по умолчанию plt.cm.bwr).
        - base_pos_colormap (Colormap): Колоркарта для положительных значений (по умолчанию plt.cm.Reds).
        - base_neg_colormap (Colormap): Колоркарта для отрицательных значений (по умолчанию plt.cm.Blues).
        - base_radius (float): Базовый радиус кругов для узлов (по умолчанию 6).
        - base_colormap (Colormap): Колоркарта для узлов (по умолчанию plt.cm.Reds).
        - fill (bool): Нужно ли заполнять круги узлов (по умолчанию True).
    """
    
    if ax is None:
        fig, ax = plt.subplots()

    base_fontisize = kwargs.get('base_fontisize', 10)

    base_alpha = kwargs.get('base_alpha', 0.2)
    base_lwidth = kwargs.get('base_lwidth', 0.3)
    base_linestyle = kwargs.get('base_linestyle', '-')
    base_bicolormap = kwargs.get('base_bicolormap', plt.cm.bwr)
    base_pos_colormap = kwargs.get('base_pos_colormap', plt.cm.Reds)
    base_neg_colormap = kwargs.get('base_neg_colormap', plt.cm.Blues)

    base_circle_radius = kwargs.get('base_radius', 6)
    base_colormap = kwargs.get('base_colormap', plt.cm.Reds)
    fill_or_not = kwargs.get('fill', True)

    # Отображение изображения
    image = plt.imread(img_path)
    ax.imshow(image)

    if len(chan_connections) > 0:
        # Определение диапазона значений в chan_connections
        chan_values = np.array(list(chan_connections.values()))
        min_val_chans = np.min(chan_values)
        max_val_chans = np.max(chan_values)
        norm_for_lines = plt.Normalize(vmin=min_val_chans, vmax=max_val_chans)

        # Рисование линий между каналами
        for key, val in chan_connections.items():
            k = tuple(key.split('-'))
            chan1 = chans_coords[k[0]]
            chan2 = chans_coords[k[1]]

            x_values = [chan1[0], chan2[0]]
            y_values = [chan1[1], chan2[1]]

            # Корретировка цвета линий, прозрачности и ширины в зависимости от диапазона значений
            if min_val_chans < 0 and max_val_chans > 0:
                color_map = base_bicolormap
            elif min_val_chans >= 0:
                color_map = base_pos_colormap
            else:
                color_map = base_neg_colormap

            color_val = norm_for_lines(val)
            line_color = color_map(color_val)
            alpha = base_alpha * abs(val) if base_alpha * abs(val) < 1 else 1
            linewidth = base_lwidth * abs(val)

            ax.plot(x_values, y_values,
                    linestyle=base_linestyle,
                    color=line_color,
                    linewidth=linewidth,
                    alpha=alpha)
            
    
    if len(node_degree) > 0:
        node_values = np.array(list(node_degree.values()))
        min_val_nodes = np.min(node_values)
        max_val_nodes = np.max(node_values)
        norm_for_circles = plt.Normalize(vmin=min_val_nodes, vmax=max_val_nodes)


        # Рисует узлы с различными размерами или цветами на основе значений степени узла
        for key, val in node_degree.items():
            x, y = chans_coords[key]

            color = base_colormap(norm_for_circles(val))
            alpha = base_alpha * abs(val) if base_alpha * abs(val) < 1 else 1
            radius = base_circle_radius * abs(val)

            circle = plt.Circle((x, y),
                                radius=radius,
                                color=color,
                                fill=fill_or_not,
                                alpha=alpha)

            ax.add_patch(circle)

    ax.set_title(frequency, fontsize=base_fontisize)
