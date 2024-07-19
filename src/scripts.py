import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import funcs


def get_libs_paths(path_to_dir, sort=True):
    """
    Возвращает словарь с путями к файлам в директории.

    Параметры:
    - path_to_dir (str): Путь к директории.
    - sort (bool): Сортировать ли результаты.

    Возвращает:
    - dict: Словарь вида {имя файла: путь}.
    """
    df_paths = {}

    for filename in os.listdir(path_to_dir):
        if filename.endswith('.xlsx'):
            key = os.path.splitext(filename)[0]
            df_paths[key] = os.path.join(path_to_dir, filename)

    if sort:
        df_paths = dict(sorted(df_paths.items()))

    return df_paths


def draw_plots(TIME_PERIODS: list,
               FREQUENCIES: list,
               df_paths: dict,
               CHANS_COORDS: dict,
               IMG_PATH: str,
               which_freq_to_draw=None,
               TRESH=0.005,
               savefig=False,
               **kwargs):
    """
    Создает и отображает графики для всех наборов данных, указанных в df_paths.

    Параметры:
    - TIME_PERIODS (list): Список временных периодов.
    - FREQUENCIES (list): Список частот.
    - df_paths (dict): Словарь с путями к файлам данных.
    - CHANS_COORDS (dict): Координаты каналов.
    - IMG_PATH (str): Путь к изображению.
    - which_freq_to_draw (tuple, optional): Кортеж с диапазоном частот для рисования.
    - TRESH (float, optional): Пороговое значение для фильтрации.
    - savefig: Флаг для сохранения полученного изображения, по умолчанию False.
    - **kwargs: Дополнительные параметры для настройки графиков.

    Ключи для графика и их значения по умолчанию:
    figname = 'plot.png' (путь для сохранения изображения),
    dpi = 400,
    fontsize = 11,
    subplot_fontsize = 9
    """
    figname = kwargs.get('figname', 'plot.png')
    dpi = kwargs.get('dpi', 400)
    fontsize = kwargs.get('fontsize', 11)
    subplot_fontsize = kwargs.get('subplot_fontsize', 9)

    if which_freq_to_draw:
        plots_in_a_row = len(range(which_freq_to_draw[0],
                                   which_freq_to_draw[1]))
    else:
        plots_in_a_row = len(FREQUENCIES)

    fig, axs = plt.subplots(len(df_paths),
                            plots_in_a_row,
                            figsize=(10, 16))
    fig.suptitle(f'T-статистика и степени узлов\np-значение={TRESH}\n',
                 fontsize=fontsize)

    if len(TIME_PERIODS) == 1:
        axs = np.expand_dims(axs, axis=0)
    if plots_in_a_row == 1:
        axs = np.expand_dims(axs, axis=1)

    for i, dataset in enumerate(df_paths):
        df = pd.read_excel(df_paths[dataset])
        assert 'Unnamed: 0' in list(df.columns), \
            'Колонка "Unnamed: 0" не найдена, назовите колонку с подключениями каналов "Unnamed: 0"'

        df.rename(columns={'Unnamed: 0': 'chan_connections'}, inplace=True)

        if which_freq_to_draw:
            df = df.iloc[:, [0] + list(range(which_freq_to_draw[0],
                                             which_freq_to_draw[1]))]

            freq_ls = list(range(which_freq_to_draw[0]-1,
                                 which_freq_to_draw[1]-1))
        else:
            df = df.iloc[:, [0] + list(range(1, plots_in_a_row+1))]
            freq_ls = list(range(plots_in_a_row))

        tmp_dict = funcs.get_chans_connection(df,
                                              thresh=TRESH,
                                              df_path_or_name=dataset)

        for j, freq in enumerate(tmp_dict):
            connects_counts, connects_weights = funcs.count_connections(tmp_dict[freq],
                                                                        calculate_stats=True)

            funcs.draw_network(chans_coords=CHANS_COORDS,
                               img_path=IMG_PATH,
                               chan_connections=tmp_dict[freq],
                               node_degree=connects_weights,
                               frequency=FREQUENCIES[freq_ls[j]],
                               ax=axs[i, j],
                               base_fontsize=subplot_fontsize)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
        axs[i, 0].set_ylabel(TIME_PERIODS[i], fontsize=fontsize)

    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    plt.tight_layout()
    if savefig:
        plt.savefig(figname, dpi=dpi)
    plt.show()


def draw_plot_one_tab(path: str,
                      name: str,
                      IMG_PATH: str,
                      CHANS_COORDS: dict,
                      TRESH=0.005,
                      which_columns=None,  # список столбцов, начиная с 1
                      vertical_layout=False,
                      tight_layout=True, 
                      savefig=False, 
                      figname='plot.png'):
    """
    Создает и отображает графики для одного набора данных.

    Параметры:
    - path (str): Путь к файлу данных.
    - name (str): Имя временного периода.
    - IMG_PATH (str): Путь к изображению.
    - CHANS_COORDS (dict): Координаты каналов.
    - TRESH (float, optional): Пороговое значение для фильтрации.
    - which_columns (list, optional): Список столбцов для рисования.
    - vertical_layout (bool, optional): Использовать ли вертикальное расположение графиков.
    - tight_layout (bool, optional): Использовать ли плотную компоновку графиков.
    - savefig: Флаг для сохранения полученного изображения, по умолчанию False.
    - figname: Наименование (путь) полученного изображения, по умолчанию 'plot.png'.
    """
    df = pd.read_excel(path)
    assert 'Unnamed: 0' in list(df.columns), \
        'Колонка "Unnamed: 0" не найдена, назовите колонку с подключениями каналов "Unnamed: 0"'

    df.rename(columns={'Unnamed: 0': 'chan_connections'}, inplace=True)

    tmp_dict = funcs.get_chans_connectionv2(df, thresh=TRESH,
                                            df_path_or_name=path,
                                            which_columns=which_columns)

    freqs = list(tmp_dict.keys())

    if vertical_layout:
        fig, axs = plt.subplots(len(freqs), 1, figsize=(8, 24))
    else:
        fig, axs = plt.subplots(1, len(freqs), figsize=(20, 6))

    fig.suptitle(f'T-статистика и степени узлов\np-значение={TRESH}\nвременной период: {name}',
                 fontsize=12)

    for j, freq in enumerate(tmp_dict):

        connects_counts, connects_weights = funcs.count_connections(tmp_dict[freq],
                                                                    calculate_stats=True)

        funcs.draw_network(chans_coords=CHANS_COORDS,
                           img_path=IMG_PATH,
                           chan_connections=tmp_dict[freq],
                           node_degree=connects_weights,
                           frequency=freqs[j],
                           ax=axs[j],
                           base_fontsize=12)
        axs[j].set_xticks([])
        axs[j].set_yticks([])

    plt.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)
    if tight_layout:
        plt.tight_layout()
    if savefig:
        plt.savefig(figname)
    plt.show()
