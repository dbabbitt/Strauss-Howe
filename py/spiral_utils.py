
#!/usr/bin/env python
# Utility Functions to initialize Strauss-Howe shapes.
# Dave Babbitt <dave.babbitt@gmail.com>
# Author: Dave Babbitt, Data Scientist
# coding: utf-8

# Soli Deo gloria

"""
self: A set of utility functions common to spiral visualization
"""
from PIL import Image, ImageDraw, ImageFont
from cycler import cycler
from datetime import date, datetime, timedelta
from io import BytesIO
from itertools import combinations
from math import cos, sin, pi, sqrt, atan, tan
from matplotlib.pyplot import imshow
from pathlib import Path
import imageio
import logging
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import re
import requests
import shutil
import webcolors
import warnings
warnings.filterwarnings('ignore')

class StraussHoweUtilities(object):
    """This class implements the core of the utility functions
    needed to create patriline spirals.
    
    Examples
    --------
    
    >>> import spiral_utils
    >>> u = spiral_utils.StraussHoweUtilities()
    """
    
    def __init__(self, s=None, verbose=False):
        if s is None:
            from storage import Storage
            self.s = Storage()
        else:
            self.s = s
        
        # Get datasets
        if self.s.pickle_exists('archetypes_df'):
            self.archetypes_df = self.s.load_object('archetypes_df')
        if self.s.pickle_exists('dresses_file_dict'):
            self.dresses_file_dict = self.s.load_object('dresses_file_dict')
        if self.s.pickle_exists('eras_df'):
            self.eras_df = self.s.load_object('eras_df')
        if self.s.pickle_exists('generations_df'):
            self.generations_df = self.s.load_object('generations_df')
        if self.s.pickle_exists('history_radius_dict'):
            self.history_radius_dict = self.s.load_object('history_radius_dict')
        if self.s.pickle_exists('history_year_dict'):
            self.history_year_dict = self.s.load_object('history_year_dict')
        if self.s.pickle_exists('patriline_df'):
            self.patriline_df = self.s.load_object('patriline_df')
        if self.s.pickle_exists('saecula_df'):
            self.saecula_df = self.s.load_object('saecula_df')
        if self.s.pickle_exists('saeculum_cmap_dict'):
            self.saeculum_cmap_dict = self.s.load_object('saeculum_cmap_dict')
        if self.s.pickle_exists('turnings_df'):
            self.turnings_df = self.s.load_object('turnings_df')
        if self.s.pickle_exists('turning_numbers_df'):
            self.turning_numbers_df = self.s.load_object('turning_numbers_df')
        if self.s.pickle_exists('us_presidents_df'):
            self.us_presidents_df = self.s.load_object('us_presidents_df')
        self.min_year = self.patriline_df.year_of_birth.min()
        self.max_year = self.patriline_df.year_of_birth.max()
        
        # URL and file path patterns
        self.url_regex = re.compile(r'\b(https?|file)://[-A-Z0-9+&@#/%?=~_|$!:,.;]*[A-Z0-9+&@#/%=~_|$]', re.IGNORECASE)
        self.filepath_regex = re.compile(r'\b[c-d]:\\(?:[^\\/:*?"<>|\x00-\x1F]{0,254}[^.\\/:*?"<>|\x00-\x1F]\\)*(?:[^\\/:*?"<>|\x00-\x1F]{0,254}[^.\\/:*?"<>|\x00-\x1F])', re.IGNORECASE)
        
        # Create movie folders
        self.jpg_dir = os.path.join(self.s.saves_folder, 'jpg')
        self.png_folder = os.path.join(self.s.saves_folder, 'png')
        self.movie_folder = os.path.join(self.s.saves_folder, 'movies')
        os.makedirs(name=self.movie_folder, exist_ok=True)
        self.temp_movie_folder = os.path.join(self.movie_folder, 'temp')
        os.makedirs(name=self.temp_movie_folder, exist_ok=True)
        self.bare_movie_folder = os.path.join(self.movie_folder, 'bare')
        os.makedirs(name=self.bare_movie_folder, exist_ok=True)
        self.saeculum_dir = os.path.join(self.s.data_folder, 'saeculum')
        os.makedirs(name=self.saeculum_dir, exist_ok=True)
        self.saeculum_movie_folder = os.path.join(self.movie_folder, 'saeculum')
        os.makedirs(name=self.saeculum_movie_folder, exist_ok=True)
        self.saeculum_fashionable_movie_folder = os.path.join(self.movie_folder, 'saeculum_fashionable')
        os.makedirs(name=self.saeculum_fashionable_movie_folder, exist_ok=True)
        
        # Color values
        self.full_corner_list = ['white', 'black', 'red', 'green', 'blue',
                                 'magenta', 'yellow', 'cyan']
        self.white_tuple = (255, 255, 255, 0)
        self.black_tuple = (0, 0, 0, 255)
        self.kryg_face_set_list = self.get_face_set_list(['black', 'red', 'yellow', 'green'])
        self.krmb_face_set_list = self.get_face_set_list(['black', 'red', 'magenta', 'blue'])
        self.kbcg_face_set_list = self.get_face_set_list(['black', 'blue', 'cyan', 'green'])
        self.wcgy_face_set_list = self.get_face_set_list(['white', 'cyan', 'green', 'yellow'])
        self.wcbm_face_set_list = self.get_face_set_list(['white', 'cyan', 'blue', 'magenta'])
        self.wyrm_face_set_list = self.get_face_set_list(['white', 'yellow', 'red', 'magenta'])
        
        # Diagram values
        self.now_year = datetime.now().year
    
    def empty_temp_folder(self, temp_folder=None, verbose=False):
        if temp_folder is None:
            temp_folder = self.temp_movie_folder
        for file_name in os.listdir(temp_folder):
            file_path = os.path.join(temp_folder, file_name)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                if verbose:
                    print('Failed to delete {file_path}. Reason: {str(e).strip()}')
    
    def show_generation_blurb(self, generation_name):
        if str(generation_name) != 'nan':
            print('{}'.format(generation_name))
            mask_series = (self.generations_df.index == generation_name[:-1])
            turnings_archetype_list = self.generations_df[mask_series].turnings_archetype.tolist()
            if not len(turnings_archetype_list):
                mask_series = (self.generations_df.index == generation_name)
                turnings_archetype_list = self.generations_df[mask_series].turnings_archetype.tolist()
            if len(turnings_archetype_list):
                turnings_archetype = turnings_archetype_list[0].lower()
                print('({})'.format(turnings_archetype))
            generations_archetype_list = self.generations_df[mask_series].generations_archetype.tolist()
            if len(generations_archetype_list):
                generations_archetype = generations_archetype_list[0].lower()
                print('{}'.format(generations_archetype))
    
    def print_turnings(self):
        for turning_name, row_series in self.turnings_df.iterrows():
            turning_year_begin = row_series.turning_year_begin
            turning_year_end = row_series.turning_year_end
            turning_notes = row_series.turning_notes
            entering_elderhood = row_series.entering_elderhood
            entering_midlife = row_series.entering_midlife
            entering_young_adulthood = row_series.entering_young_adulthood
            entering_childhood = row_series.entering_childhood
            print()
            print('{}-{}'.format(turning_year_begin, turning_year_end))
            print('{}'.format('\n'.join(turning_notes.split('. '))))
            print('-------------------------')
            self.show_generation_blurb(entering_elderhood)
            print('-------------------------')
            self.show_generation_blurb(entering_midlife)
            print('-------------------------')
            self.show_generation_blurb(entering_young_adulthood)
            print('-------------------------')
            self.show_generation_blurb(entering_childhood)
            print('-------------------------')
    
    def convert_years_to_x(self, year, y_tuple=(0, 2220), years_tuple=(1435, 2029)):
        a = (y_tuple[1]-y_tuple[0])/(years_tuple[1]-years_tuple[0])
        b = y_tuple[0] - (years_tuple[0]*a)
        x = a*year + b
        
        return x
    
    def convert_age_to_y(self, age, size_tuple=(333, 133), age_tuple=(20, 80)):
        a = (size_tuple[0] - size_tuple[1])/(age_tuple[0] - age_tuple[1])
        b = size_tuple[0] - (age_tuple[0]*a)
        y = a*age + b
        
        return y
    
    def draw_hl(self, draw, y, image_width):
        draw.line(xy=(0, y, image_width, y), fill=(255, 0, 0), width=1, joint=None)
    
    def add_fashion_image(self, year):
        
        # Get old image data
        old_path = os.path.join(self.png_folder, 'plot_{}.png'.format(year))
        foreground = Image.open(old_path)
        foreground = foreground.convert('RGBA')
        old_data_list = foreground.getdata()
        
        # Get new image data
        new_data_list = []
        for old_tuple in old_data_list:
            if (old_tuple[0] == 255) and (old_tuple[1] == 255) and (old_tuple[2] == 255):
                new_data_list.append(self.white_tuple)
            else:
                new_data_list.append(old_tuple)
        
        # Replace old with new
        foreground.putdata(new_data_list)
        
        # Get dresses image
        dresses_file = self.dresses_file_dict[year]
        if dresses_file is None:
            file_path = os.path.join(self.png_folder, 'plot_{}_fashionable.png'.format(year))
            foreground.save(file_path, 'PNG')
        else:
            file_path = os.path.join(self.png_folder, dresses_file)
            dresses_img = Image.open(file_path)
            dresses_img = dresses_img.convert('RGBA')
            
            dresses_img.paste(foreground, (0, 0), foreground)
            
            file_path = os.path.join(self.png_folder, 'plot_{}_fashionable.png'.format(year))
            dresses_img.save(file_path, 'PNG')
    
    def add_fashion_saeculum_image(self, year):
        
        # Get old image data
        old_path = os.path.join(self.bare_movie_folder, 'plot_{}.png'.format(year))
        foreground = Image.open(old_path)
        foreground = foreground.convert('RGBA')
        foreground = self.make_transparent(foreground, threshold=6)
        
        # Get dresses image
        dresses_file = self.dresses_file_dict[year]
        if dresses_file is None:
            dresses_img = foreground
        else:
            file_path = os.path.join(self.png_folder, dresses_file)
            dresses_img = Image.open(file_path)
            dresses_img = dresses_img.convert('RGBA')
            dresses_img = self.make_transparent(dresses_img, threshold=5)
            dresses_img.paste(foreground, (0, 0), foreground)
        
        # Get saeculum image
        mask_series = (self.turnings_df.turning_year_begin <= year) & (self.turnings_df.turning_year_end >= year)
        saeculum_list = self.turnings_df[mask_series].index.tolist()
        if len(saeculum_list):
            saeculum_name = saeculum_list[0]
            saeculum_file = '{}.png'.format(saeculum_name)
            saeculum_path = Path(os.path.join(self.saeculum_dir, saeculum_file))
            if saeculum_path.is_file():
                saeculum_img = Image.open(saeculum_path, mode='r')
                saeculum_img = saeculum_img.convert('RGBA')
                saeculum_img.paste(dresses_img, (0, 0), dresses_img)
            else:
                saeculum_img = dresses_img
            file_path = os.path.join(self.saeculum_fashionable_movie_folder, 'plot_{}_saeculum_fashionable.png'.format(year))
            saeculum_img.save(file_path, 'PNG')
    
    def add_saeculum_image(self, year):
        
        # Get old image data
        old_path = os.path.join(self.bare_movie_folder, 'plot_{}.png'.format(year))
        foreground = Image.open(old_path)
        foreground = foreground.convert('RGBA')
        old_data_list = foreground.getdata()
        
        # Get new image data
        new_data_list = []
        for old_tuple in old_data_list:
            if (old_tuple[0] == 255) and (old_tuple[1] == 255) and (old_tuple[2] == 255):
                new_data_list.append(self.white_tuple)
            else:
                new_data_list.append(old_tuple)
        
        # Replace old with new
        foreground.putdata(new_data_list)
        
        # Get saeculum image
        mask_series = (self.turnings_df.turning_year_begin <= year) & (self.turnings_df.turning_year_end >= year)
        saeculum_list = self.turnings_df[mask_series].index.tolist()
        if len(saeculum_list):
            saeculum_name = saeculum_list[0]
            saeculum_file = '{}.png'.format(saeculum_name)
            saeculum_path = Path(os.path.join(self.saeculum_dir, saeculum_file))
            new_path = os.path.join(self.saeculum_movie_folder, 'plot_{}_saeculum.png'.format(year))
            if saeculum_path.is_file():
                saeculum_img = Image.open(saeculum_path, mode='r')
                saeculum_img = saeculum_img.convert('RGBA')
                saeculum_img.paste(foreground, (0, 0), foreground)
                saeculum_img.save(new_path, 'PNG')
            else:
                foreground.save(new_path, 'PNG')
    
    def add_dalle_background(self, year, movie_folder=None):
        if movie_folder is None:
            movie_folder = self.temp_movie_folder
        
        # Get old image data
        old_path = os.path.join(movie_folder, f'plot_{year}.png')
        background = Image.open(old_path)
        background = background.convert('RGBA')
        old_data_list = background.getdata()
        
        # Get new image data
        new_data_list = []
        for old_tuple in old_data_list:
            if (old_tuple[0] == 255) and (old_tuple[1] == 255) and (old_tuple[2] == 255):
                new_data_list.append(self.white_tuple)
            else:
                new_data_list.append(old_tuple)
        
        # Replace old with new
        background.putdata(new_data_list)
        
        # Get DALL·E image
        file_path = os.path.join(self.s.data_folder, 'png', 'dall_e_cover.png')
        dall_e_img = Image.open(file_path)
        dall_e_img = dall_e_img.convert('RGBA')
        
        dall_e_img.paste(background, (0, 0), background)
        
        file_path = os.path.join(movie_folder, 'plot_{}_dall_e.png'.format(year))
        dall_e_img.save(file_path, 'PNG')
        
        return file_path
    
    def polar_to_cartesian(self, r, theta):
        radians = theta*(pi/180)
        
        return int(r*cos(radians)), int(r*sin(radians))
    
    def add_spiral_labels(
        self, years_list, history_year_dict, theta_offset=0, i=0, ax=None
    ):
        if ax is None:
            ax = plt.gca()
        i = i % 4
        for year in years_list:
            radius, theta = history_year_dict[year]
            theta += theta_offset
            radius += 25*i
            radius -= 25/2
            x, y = self.polar_to_cartesian(radius, theta)
            text_obj = ax.text(
                x, y, year, fontsize=10, color='gray',
                rotation=theta-90, rotation_mode='anchor'
            )

    def min_max_norm(self, raw_list, verbose=False):
        norm_list = []
        min_value = min(raw_list)
        max_value = max(raw_list)
        
        for value in raw_list:
            normed_value = (value - min_value) / (max_value - min_value)
            norm_list.append(normed_value)
        
        return norm_list
    
    def adjust_axis(self, ax=None, verbose=False):
        if ax is None:
            ax = plt.gca()
        ax.axis('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for border in ['top', 'right', 'bottom', 'left']:
            ax.spines[border].set_visible(False)
    
    def add_theta_labels(
        self, divisor=64, years_list=None, history_year_dict=None,
        bottom_year=None, ax=None, verbose=False
    ):
        if ax is None:
            ax = plt.gca()
        x_list = []
        y_list = []
        def append_lists(theta):
            x, y = self.polar_to_cartesian(128, theta)
            x_list.append(x)
            y_list.append(y)
            thetas_list.append(theta)
        if history_year_dict is None:
            addend = 360//divisor
            theta = 0 - addend
            thetas_list = []
            while theta < (360 - addend):
                theta += addend
                append_lists(theta)
            labels_list = thetas_list
        else:
            thetas_list = []
            labels_list = []
            theta_offset = 0
            if (bottom_year is not None):
                if (bottom_year in history_year_dict.keys()):
                    theta_offset = 270 - history_year_dict[bottom_year][1]
            if years_list is None:
                for year, (_, theta) in history_year_dict.items():
                    if isinstance(theta, (int, float)):
                        append_lists(theta+theta_offset)
                        labels_list.append(year)
            else:
                for year in years_list:
                    _, theta = history_year_dict[year]
                    if isinstance(theta, (int, float)):
                        append_lists(theta+theta_offset)
                        labels_list.append(year)
        for x, y, theta, label in zip(
            self.min_max_norm(x_list),
            self.min_max_norm(y_list),
            thetas_list,
            labels_list
        ):
            if verbose:
                print(x, y, theta, label)
            if (theta % 90):
                text_obj = ax.text(
                    x, y, label, fontsize=10, color='gray', ha='center',
                    va='center', transform=ax.transAxes
                )
            else:
                text_obj = ax.text(
                    x, y, label, fontsize=10, color='green', ha='center',
                    va='center', transform=ax.transAxes, weight='bold'
                )
        self.adjust_axis(ax=ax)
    
    def add_patriarch_label(
        self, patriarch_name, history_year_dict, bottom_year,
        theta_offset=0, i=0, ax=None
    ):
        if ax is None:
            ax = plt.gca()
        i = i % 4
        radius, theta = history_year_dict[bottom_year]
        theta += theta_offset
        radius += 25*i
        radius -= 25/2
        x, y = self.polar_to_cartesian(radius, theta)
        text_obj = ax.text(
            x, y, patriarch_name, fontsize=8, color='gray',
            ha='center', va='center'
        )
    
    def archimedes_spiral(self, theta, theta_offset=0.0):
        """
        Return Archimedes spiral
        
        Args:
            theta: array-like, angles from polar coordinates to be converted
            theta_offset: float, angle offset in radians (2*pi = 0)
        """
        
        (x, y) = (theta * np.cos(theta + theta_offset), theta
                  * np.sin(theta + theta_offset))
        x_norm = np.max(np.abs(x))
        y_norm = np.max(np.abs(y))
        (x, y) = (x / x_norm, y / y_norm)
        
        return (x, y)
    
    def bernoulli_spiral(self, theta, theta_offset=0.0, *args, **kwargs):
        """
        Return Equiangular (Bernoulli's) spiral
        
        Args:
        theta: array-like, angles from polar coordinates to be converted
        theta_offset: float, angle offset in radians (2*pi = 0)
        
        Kwargs:
        exp_scale: growth rate of the exponential
        """
        
        exp_scale = kwargs.pop('exp_scale', 0.1)
        
        (x, y) = (np.exp(exp_scale * theta) * np.cos(theta + theta_offset),
                  np.exp(exp_scale * theta) * np.sin(theta + theta_offset))
        x_norm = np.max(np.abs(x))
        y_norm = np.max(np.abs(y))
        (x, y) = (x / x_norm, y / y_norm)
        
        return (x, y)
    
    def colors_dict_to_dataframe(self, colors_dict):
        columns_list = ['Red', 'Green', 'Blue']
        rows_list = []
        index_list = []
        for base_name, color_tuple in colors_dict.items():
            row_dict = {}
            index_list.append(base_name)
            for i, color_value in enumerate(columns_list):
                row_dict[color_value] = color_tuple[i]
            rows_list.append(row_dict)
        df = pd.DataFrame(rows_list, columns=columns_list, index=index_list)
        
        return df
    
    def conjunctify_nouns(self, noun_list):
        if len(noun_list) > 2:
            list_str = ', and '.join([', '.join(noun_list[:-1])] + [noun_list[-1]])
        elif len(noun_list) == 2:
            list_str = ' and '.join(noun_list)
        elif len(noun_list) == 1:
            list_str = noun_list[0]
        else:
            list_str = ''
        
        return list_str
    
    def create_xy_list(self, history_radius_dict):
        xy_list = []
        for radius in sorted(history_radius_dict.keys()):
            year, theta = history_radius_dict[radius]
            cartesian_tuple = self.polar_to_cartesian(radius, theta)
            if len(xy_list):
                if (cartesian_tuple != xy_list[-1]):
                    xy_list.append(cartesian_tuple)
            else:
                xy_list.append(cartesian_tuple)
        
        return xy_list
    
    def display_test_colors(
        self, test_list, saeculum_title, face_title, nearness_str='far from', color_dict=mcolors.XKCD_COLORS,
        color_title='XKCD', face_point='Face', verbose=True
    ):
        if verbose: print(f'test_list = "{test_list}"')
        name_list = [name for distance, name in test_list]
        colors_dict = {name: color for name, color in color_dict.items() if name in name_list}
        title_str = '{} {} Colors, {} the {} {}'.format(color_title, saeculum_title, nearness_str,
                                                        face_title, face_point)
        self.plot_colortable(colors_dict=colors_dict, title=title_str, sort_colors=True, emptycols=0)
    
    def distance_between(self, new_tuple, old_tuple):
        green_diff = new_tuple[0] - old_tuple[0]
        blue_diff = new_tuple[1] - old_tuple[1]
        red_diff = new_tuple[2] - old_tuple[2]
        
        return sqrt(green_diff**2 + blue_diff**2 + red_diff**2)
    
    def distance_from_black(self, old_tuple):
        
        return sqrt(old_tuple[0]**2 + old_tuple[1]**2 + old_tuple[2]**2)
    
    def distance_from_blue(self, old_tuple):
        blue_diff = 1.0 - old_tuple[1]
        
        return sqrt(old_tuple[0]**2 + blue_diff**2 + old_tuple[2]**2)
    
    def distance_from_cyan(self, old_tuple):
        green_diff = 1.0 - old_tuple[0]
        blue_diff = 1.0 - old_tuple[1]
        
        return sqrt(green_diff**2 + blue_diff**2 + old_tuple[2]**2)
    
    def distance_from_green(self, old_tuple):
        green_diff = 1.0 - old_tuple[0]
        
        return sqrt(green_diff**2 + old_tuple[1]**2 + old_tuple[2]**2)
    
    def distance_from_kbcg_face(self, old_tuple):
        green_diff = 0.5 - old_tuple[0]
        blue_diff = 0.5 - old_tuple[1]
        
        return sqrt(green_diff**2 + blue_diff**2 + old_tuple[2]**2)
    
    def distance_from_krmb_face(self, old_tuple):
        blue_diff = 0.5 - old_tuple[1]
        red_diff = 0.5 - old_tuple[2]
        
        return sqrt(old_tuple[0]**2 + blue_diff**2 + red_diff**2)
    
    def distance_from_kryg_face(self, old_tuple):
        green_diff = 0.5 - old_tuple[0]
        red_diff = 0.5 - old_tuple[2]
        
        return sqrt(green_diff**2 + old_tuple[1]**2 + red_diff**2)
    
    def distance_from_magenta(self, old_tuple):
        blue_diff = 1.0 - old_tuple[1]
        red_diff = 1.0 - old_tuple[2]
        
        return sqrt(old_tuple[0]**2 + blue_diff**2 + red_diff**2)
    
    def distance_from_red(self, old_tuple):
        red_diff = 1.0 - old_tuple[2]
        
        return sqrt(old_tuple[0]**2 + old_tuple[1]**2 + red_diff**2)
    
    def distance_from_wcbm_face(self, old_tuple):
        green_diff = 0.5 - old_tuple[0]
        blue_diff = 1.0 - old_tuple[1]
        red_diff = 0.5 - old_tuple[2]
        
        return sqrt(green_diff**2 + blue_diff**2 + red_diff**2)
    
    def distance_from_wcgy_face(self, old_tuple):
        green_diff = 1.0 - old_tuple[0]
        blue_diff = 0.5 - old_tuple[1]
        red_diff = 0.5 - old_tuple[2]
        
        return sqrt(green_diff**2 + blue_diff**2 + red_diff**2)
    
    def distance_from_white(self, old_tuple):
        green_diff = 1.0 - old_tuple[0]
        blue_diff = 1.0 - old_tuple[1]
        red_diff = 1.0 - old_tuple[2]
        
        return sqrt(green_diff**2 + blue_diff**2 + red_diff**2)
    
    def distance_from_wyrm_face(self, old_tuple):
        green_diff = 0.5 - old_tuple[0]
        blue_diff = 0.5 - old_tuple[1]
        red_diff = 1.0 - old_tuple[2]
        
        return sqrt(green_diff**2 + blue_diff**2 + red_diff**2)
    
    def distance_from_yellow(self, old_tuple):
        green_diff = 1.0 - old_tuple[0]
        red_diff = 1.0 - old_tuple[2]
        
        return sqrt(green_diff**2 + old_tuple[1]**2 + red_diff**2)
    
    def translate_upper_left_to_center(self, point, screen_size):
        """
        Takes a point and converts it to the appropriate coordinate system.
        Note that PIL uses upper left as 0, we want the center.
        Args:
            point (real, real): A point in space.
            screen_size (int): Size of an N x N screen.
        Returns:
            (real, real): Translated point for Pillow coordinate system.
        """
        
        return point[0] + screen_size / 2, point[1] + screen_size / 2
    
    def exists(self, path):
        r = requests.head(path)
        
        return r.status_code == requests.codes.ok
    
    def fermat_spiral(self, theta, theta_offset=0.0):
        """
        Return Parabolic (Fermat's) spiral
        
        Args:
            theta: array-like, angles from polar coordinates to be converted
            theta_offset: float, angle offset in radians (2*pi = 0)
        """
        
        (x, y) = (np.sqrt(theta) * np.cos(theta + theta_offset),
                  np.sqrt(theta) * np.sin(theta + theta_offset))
        x_norm = np.max(np.abs(x))
        y_norm = np.max(np.abs(y))
        (x, y) = (x / x_norm, y / y_norm)
        
        return (x, y)
    
    def get_distance_dataframe(self, colors_df, color_title='XKCD'):
        rows_list = []
        columns_list = ['color_title', 'distance_from_white', 'distance_from_black',
                        'distance_from_red', 'distance_from_green', 'distance_from_blue',
                        'distance_from_magenta', 'distance_from_yellow', 'distance_from_cyan',
                        'distance_from_kryg_face', 'distance_from_krmb_face', 'distance_from_kbcg_face',
                        'distance_from_wcgy_face', 'distance_from_wcbm_face', 'distance_from_wyrm_face']
        index_list = []
        for row_index, row_series in colors_df.iterrows():
            green_value = row_series.Green
            blue_value = row_series.Blue
            red_value = row_series.Red
            row_tuple = (green_value, blue_value, red_value)
            row_dict = {}
            row_dict['color_title'] = color_title
            row_dict['distance_from_white'] = self.distance_from_white(row_tuple)
            row_dict['distance_from_black'] = self.distance_from_black(row_tuple)
            row_dict['distance_from_red'] = self.distance_from_red(row_tuple)
            row_dict['distance_from_green'] = self.distance_from_green(row_tuple)
            row_dict['distance_from_blue'] = self.distance_from_blue(row_tuple)
            row_dict['distance_from_magenta'] = self.distance_from_magenta(row_tuple)
            row_dict['distance_from_yellow'] = self.distance_from_yellow(row_tuple)
            row_dict['distance_from_cyan'] = self.distance_from_cyan(row_tuple)
            row_dict['distance_from_kryg_face'] = self.distance_from_kryg_face(row_tuple)
            row_dict['distance_from_krmb_face'] = self.distance_from_krmb_face(row_tuple)
            row_dict['distance_from_kbcg_face'] = self.distance_from_kbcg_face(row_tuple)
            row_dict['distance_from_wcgy_face'] = self.distance_from_wcgy_face(row_tuple)
            row_dict['distance_from_wcbm_face'] = self.distance_from_wcbm_face(row_tuple)
            row_dict['distance_from_wyrm_face'] = self.distance_from_wyrm_face(row_tuple)
            rows_list.append(row_dict)
            index_list.append(row_index)
        distance_df = pd.DataFrame(rows_list, columns=columns_list, index=index_list)
        
        return distance_df
    
    def get_face_dictionary(self, distance_df):
        face_dictionary = {}
        for row_index, row_series in distance_df.iterrows():
            tuple_list = sorted(row_series.to_dict().items(), key=lambda x: x[1])
            if tuple_list[0][1] == 0.0:
                face_dictionary[row_index] = tuple_list[0][0].split('_')[2]
            else:
                corners_list = tuple_list[:3]
                face_set = set([corners_list[0][0].split('_')[2],
                                corners_list[1][0].split('_')[2],
                                corners_list[2][0].split('_')[2]])
                if face_set in self.kryg_face_set_list:
                    face_dictionary[row_index] = 'black-red-yellow-green'
                elif face_set in self.krmb_face_set_list:
                    face_dictionary[row_index] = 'black-red-magenta-blue'
                elif face_set in self.kbcg_face_set_list:
                    face_dictionary[row_index] = 'black-blue-cyan-green'
                elif face_set in self.wcgy_face_set_list:
                    face_dictionary[row_index] = 'white-cyan-green-yellow'
                elif face_set in self.wcbm_face_set_list:
                    face_dictionary[row_index] = 'white-cyan-blue-magenta'
                elif face_set in self.wyrm_face_set_list:
                    face_dictionary[row_index] = 'white-yellow-red-magenta'
                else:
                    face_dictionary[row_index] = '-'.join(list(face_set))
        
        return face_dictionary
    
    def get_face_set_list(self, combinations_list):
        combs_obj = combinations(combinations_list, 3)
        face_set_list = []
        for color_tuple in combs_obj:
            face_set_list.append(set(color_tuple))
        
        return face_set_list
    
    def get_hsv_dict(self, colors_dict):
        """
        Hue, Saturation, Value
        """
        
        return {name: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))) for name,
                color in colors_dict.items()}
    
    def get_one_arc(
        self, start_year, stop_year, history_year_dict, theta_offset=0,
        i=0, verbose=False
    ):
        i = i % 4
        if verbose:
            if start_year in history_year_dict:
                print(f'history_year_dict[start_year] = history_year_dict[{start_year}] = {history_year_dict[start_year]}')
        start_radius = history_year_dict[start_year][0]
        start_radius += 25*i
        stop_radius = history_year_dict[stop_year][0]
        stop_radius += 25*i
        increment_count = int(2*pi*start_radius)
        radius_array = np.linspace(start=start_radius, stop=stop_radius,
                                   num=increment_count)
        start_theta = history_year_dict[start_year][1]
        if verbose:
            print(f'start_theta = {start_theta}')
        stop_theta = history_year_dict[stop_year][1]
        if verbose:
            print(f'stop_theta = {stop_theta}')
        theta_array = np.linspace(start=start_theta, stop=stop_theta,
                                  num=increment_count)
        xy_list = []
        for radius, theta in zip(radius_array, theta_array):
            theta += theta_offset
            cartesian_tuple = self.polar_to_cartesian(radius, theta)
            if len(xy_list):
                if (cartesian_tuple != xy_list[-1]):
                    xy_list.append(cartesian_tuple)
            else:
                xy_list.append(cartesian_tuple)
        
        return xy_list
    
    def get_one_stopped_arc(self, start_year, stop_year, stopped_year,
                            history_year_dict, i=0):
        i = i % 4
        if stop_year > stopped_year:
            stop_year = stopped_year
        start_radius = history_year_dict[start_year][0]
        start_radius += 25*i
        stop_radius = history_year_dict[stop_year][0]
        stop_radius += 25*i
        increment_count = int(2*pi*start_radius)
        radius_array = np.linspace(start=start_radius, stop=stop_radius,
                                   num=increment_count)
        start_theta = history_year_dict[start_year][1]
        stop_theta = history_year_dict[stop_year][1]
        theta_array = np.linspace(start=start_theta, stop=stop_theta,
                                  num=increment_count)
        xy_list = []
        for radius, theta in zip(radius_array, theta_array):
            cartesian_tuple = self.polar_to_cartesian(radius, theta)
            if len(xy_list):
                if (cartesian_tuple != xy_list[-1]):
                    xy_list.append(cartesian_tuple)
            else:
                xy_list.append(cartesian_tuple)
        
        return xy_list
    
    def get_page_tables(self, url_or_filepath_or_html, driver=None, pdf_file_name=None, verbose=True):
        '''
        tables_url = 'https://en.wikipedia.org/wiki/Provinces_of_Afghanistan'
        page_tables_list = u.get_page_tables(tables_url)
        
        url = 'https://crashstats.nhtsa.dot.gov/Api/Public/Publication/812581'
        file_name = '2016_State_Traffic_Data_CrashStats_NHTSA.pdf'
        page_tables_list = u.get_page_tables(url, pdf_file_name=file_name)
        '''
        tables_df_list = []
        if pdf_file_name is not None:
            data_pdf_folder = os.path.join(self.s.data_folder, 'pdf')
            os.makedirs(name=data_pdf_folder, exist_ok=True)
            file_path = os.path.join(data_pdf_folder, pdf_file_name)
            import requests
            response = requests.get(url_or_filepath_or_html)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            import tabula
            tables_df_list = tabula.read_pdf(file_path, pages='all')
        elif self.url_regex.fullmatch(url_or_filepath_or_html) or self.filepath_regex.fullmatch(os.path.abspath(url_or_filepath_or_html)):
            from urllib.error import HTTPError
            try:
                tables_df_list = pd.read_html(url_or_filepath_or_html)
            except (ValueError, HTTPError) as e:
                if verbose: print(str(e).strip())
                page_soup = self.get_page_soup(url_or_filepath_or_html, driver=driver)
                table_soups_list = page_soup.find_all('table')
                for table_soup in table_soups_list:
                    tables_df_list += self.get_page_tables(str(table_soup), driver=None, verbose=False)
        else:
            import io
            f = io.StringIO(url_or_filepath_or_html)
            tables_df_list = pd.read_html(f)
        if verbose:
            print(sorted(
                [(i, df.shape) for (i, df) in enumerate(tables_df_list)],
                key=lambda x: x[1][0], reverse=True
            ))
        
        return tables_df_list
    
    def get_row_label(self, present_year, patriarch_name, row_series):
        patriarch_age = present_year - int(row_series.year_of_birth)
        year_of_death = row_series.year_of_death
        generation_name = row_series.generation_name
        try:
            year_of_death = int(year_of_death)
        except:
            year_of_death = present_year + 1
        age_str = ''
        if (year_of_death > present_year):
            if patriarch_age > 80:
                age_str = ' in Late Elderhood'
            elif patriarch_age > 60:
                age_str = ' in Elderhood'
            elif patriarch_age > 40:
                age_str = ' in Midlife'
            elif patriarch_age > 20:
                age_str = ' as a Young Adult'
            else:
                age_str = ' in Childhood'
        label_str = '{} ({} Generation{})'.format(patriarch_name, generation_name, age_str)
        
        return label_str
    
    def get_shortest_distance(self, row_series):
        for column_name in ['xkcd_color', 'css4_color']:
            new_column_name = '{}_text_color'.format(column_name.split('_')[0])
            color = mcolors.to_rgb(row_series[column_name])
            white_distance = self.distance_from_white(color)
            black_distance = self.distance_from_black(color)
            if min(white_distance, black_distance) == white_distance:
                row_series[new_column_name] = 'black'
            else:
                row_series[new_column_name] = 'white'
        
        return row_series
    
    def label_arc(self, start_year, stopped_year,
                  history_theta_dict, arc_label, history_year_dict, ideal_distance=13,
                  i=0, label_color='black', ax=None):
        if ax is None:
            ax = plt.gca()
        i = i % 4
        starting_year = int(((start_year + stopped_year) / 2) - (len(arc_label) / 2))
        starting_radius, starting_theta = history_year_dict[starting_year]
        next_radius, next_theta = history_year_dict[starting_year+1]
        
        # Tan(A) = Opposite/Adjacent
        radians = atan(ideal_distance/starting_radius)
        theta_sign = np.sign(next_theta-starting_theta)
        
        # Increment the theta so that it spaces the letters the same regardless of the radius
        theta_increment = theta_sign*(radians*180/pi)
        
        # Figure out if you have to flip the characters upside-down and place them in right-to-left order
        if (starting_theta%360) < 200:
            #logging.info('')
            #logging.info('Right-side up thetas:')
            theta = starting_theta
            radius = starting_radius + 25*i
            radius -= 25/2
            for c in arc_label[::int(-theta_sign)]:
                #logging.info('c: "{}", radius: "{}", theta: "{}"'.format(c, radius, theta % 360))
                x, y = self.polar_to_cartesian(radius, theta)
                text_obj = ax.text(x, y, c, fontsize=12, color=label_color,
                                    rotation=theta-90, rotation_mode='anchor')
                theta += theta_increment
                if int(theta) in history_theta_dict:
                    radius = history_theta_dict[int(theta)][1]
                    radius += 25*i
                    radius -= 25/2
        else:
            logging.info('')
            logging.info('Upside-down thetas:')
            theta = starting_theta + theta_increment*len(arc_label)
            if int(theta) in history_theta_dict:
                radius = history_theta_dict[int(theta)][1]
                radius += 25*i
            else:
                radius = starting_radius + 25*i
            radius += 25/2
            for c in arc_label[::int(-theta_sign)]:
                logging.info('c: "{}", radius: "{}", theta: "{}"'.format(c, radius, theta % 360))
                x, y = self.polar_to_cartesian(radius, theta)
                text_obj = ax.text(x, y, c, fontsize=12, color=label_color,
                                    rotation=theta+90, rotation_mode='anchor')
                theta -= theta_increment
                if int(theta) in history_theta_dict:
                    radius = history_theta_dict[int(theta)][1]
                    radius += 25*i
                    radius += 25/2
    
    def make_transparent(self, img, threshold=38):
        margin = 255/self.distance_from_white(self.black_tuple)
        old_data_list = img.getdata()
        
        # Get new image data
        new_data_list = []
        for old_tuple in old_data_list:
            transparency = int(margin * self.distance_from_white(old_tuple))
            if transparency > threshold:
                transparency = 255
            elif transparency < 0:
                transparency = 0
            old_tuple = (old_tuple[0], old_tuple[1], old_tuple[2], transparency)
            new_data_list.append(old_tuple)
        
        # Replace old with new
        img.putdata(new_data_list)
        
        return img
    
    def plot_colortable(
        self, colors_dict, title, sort_colors=True, emptycols=0, ax=None
    ):
        if len(colors_dict):
            cell_width = 212
            cell_height = 22
            swatch_width = 48
            margin = 12
            topmargin = 40
            
            # Sort colors_dict by hue, saturation, value and name.
            if sort_colors is True:
                by_hsv = sorted(
                    (tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name)
                    for name, color in colors_dict.items()
                )
                names = [name for hsv, name in by_hsv]
            else:
                names = list(colors_dict)
            
            n = len(names)
            ncols = 4 - emptycols
            nrows = n // ncols + int(n % ncols > 0)
            
            width = cell_width * 4 + 2 * margin
            height = cell_height * nrows + margin + topmargin
            dpi = 72
            
            # Create the figure and subplot
            if ax is None: fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
            else:
                
                # Set the figsize and dpi
                fig = ax.get_figure()
                fig.set_size_inches(width / dpi, height / dpi)
                fig.set_dpi(dpi)
            
            fig.subplots_adjust(margin/width, margin/height,
                                (width-margin)/width, (height-topmargin)/height)
            ax.set_xlim(0, cell_width * 4)
            ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
            ax.yaxis.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.set_axis_off()
            ax.set_title(title, fontsize=24, loc='left', pad=10)
            
            for i, name in enumerate(names):
                row = i % nrows
                col = i // nrows
                y = row * cell_height
                
                swatch_start_x = cell_width * col
                swatch_end_x = cell_width * col + swatch_width
                text_pos_x = cell_width * col + swatch_width + 7
                
                ax.text(text_pos_x, y, name, fontsize=14,
                        horizontalalignment='left',
                        verticalalignment='center')
                
                ax.hlines(y, swatch_start_x, swatch_end_x,
                          color=colors_dict[name], linewidth=18)
    
    def plot_year(
        self, history_year_dict, bottom_year, ax=None, verbose=False
    ):
        if ax is None:
            ax = plt.gca()
        theta_offset = 0
        if (bottom_year in history_year_dict.keys()):
            theta_offset = 270 - history_year_dict[bottom_year][1]
        
        birth_series = self.patriline_df.year_of_birth
        death_series = self.patriline_df.year_of_death
        mask_series = (bottom_year >= birth_series)
        patriarch_names_list = self.patriline_df[mask_series].index.tolist()
        mask_series &= (bottom_year <= death_series) | death_series.isnull()
        labels_list = self.patriline_df[mask_series].index.tolist()
        for i, patriarch_name in enumerate(patriarch_names_list):
            self.plot_patriarch(
                patriarch_name, history_year_dict, bottom_year,
                theta_offset, i=i, ax=ax,
                add_label=bool(patriarch_name in labels_list)
            )
        text_obj = ax.text(
            0.5, 0, bottom_year, fontsize=12, color='black',
            ha='center', transform=ax.transAxes
        )
        self.adjust_axis(ax=ax)
    
    def plot_patriarch(
        self, patriarch_name, history_year_dict, bottom_year,
        theta_offset=0, i=0, ax=None, add_label=True
    ):
        if ax is None:
            ax = plt.gca()
        
        mask_series = (self.patriline_df.index == patriarch_name)
        start_year = int(self.patriline_df[mask_series].year_of_birth.tolist()[0])
        stop_year = self.patriline_df[mask_series].year_of_death.tolist()[0]
        try:
            stop_year = int(stop_year)
        except:
            stop_year = max(history_year_dict.keys())
        xy_list = self.get_one_arc(
            start_year=start_year, stop_year=min(stop_year, bottom_year),
            history_year_dict=history_year_dict, theta_offset=theta_offset, i=i
        )
        css4_color = self.patriline_df[mask_series].css4_color.squeeze()
        PathCollection_obj = ax.plot(
            [x[0] for x in xy_list], [y[1] for y in xy_list],
            color=css4_color, alpha=0.5
        )
        if add_label:
            self.add_patriarch_label(
                patriarch_name=patriarch_name,
                history_year_dict=history_year_dict, bottom_year=bottom_year,
                theta_offset=theta_offset, i=i
            )
    
    def save_stopped_babbitt_plot_as_blender_script(
        self, stopped_year, out_file_path, footer_str,
        history_year_dict, py_file_header_str, verbose=False
    ):
        mask_series = (self.patriline_df.year_of_birth <= stopped_year)
        i = self.patriline_df[mask_series].shape[0]-1
        out_file_path = os.path.abspath(out_file_path)
        if verbose:
            print(f'Saving to {out_file_path}')
        Path(out_file_path).touch()
        with open(out_file_path, 'w') as output:
            size = output.write(py_file_header_str)
            for patriarch_name, row_series in self.patriline_df[mask_series].iterrows():
                start_year = int(row_series.year_of_birth)
                stop_year = row_series.year_of_death
                try:
                    stop_year = int(stop_year)
                except:
                    stop_year = start_year + 80
                    if stop_year > max(history_year_dict.keys()):
                        stop_year = max(history_year_dict.keys())
                xy_list = self.get_one_stopped_arc(
                    start_year=start_year, stop_year=stop_year,
                    stopped_year=stopped_year,
                    history_year_dict=history_year_dict, i=i
                )
                size = output.write("patriarch_coords_dict['{}'] = {}\n".format(
                    patriarch_name, str([(x,y,1) for (x, y) in xy_list])
                ))
                i -= 1
            size = output.write(footer_str)
    
    def show_babbitt_plot(
        self, history_theta_dict, history_year_dict, verbose=False, ax=None
    ):
        if ax is None:
            ax = plt.gca()
        fig = plt.figure(figsize=(13, 13))
        ax = fig.add_subplot(111, autoscale_on=False)
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        i = self.patriline_df.shape[0]-1
        d = 5
        previous_saeculum = self.patriline_df.head(1).saeculum_name.tolist()[0]
        for patriarch_name, row_series in self.patriline_df.iterrows():
            start_year = int(row_series.year_of_birth)
            stop_year = row_series.year_of_death
            try:
                stop_year = int(stop_year)
            except:
                stop_year = start_year + 80
                if stop_year > max(history_year_dict.keys()):
                    stop_year = max(history_year_dict.keys())
            xy_list = self.get_one_arc(
                start_year=start_year, stop_year=stop_year,
                history_year_dict=history_year_dict, i=i
            )
            self.add_spiral_labels(
                [start_year, stop_year], history_year_dict, i
            )
            self.label_arc(
                start_year=start_year, stopped_year=stop_year,
                history_theta_dict=history_theta_dict,
                arc_label=patriarch_name, history_year_dict=history_year_dict,
                ideal_distance=13, i=i, label_color='black'
            )
            saeculum = row_series.saeculum_name
            if saeculum != previous_saeculum:
                previous_saeculum = saeculum
                d = 5
            if verbose:
                print(f'patriarch_name, i, d, saeculum = "{patriarch_name, i, d, saeculum}"')
            cmap = self.saeculum_cmap_dict[saeculum]
            c = plt.get_cmap(cmap)(np.linspace(0, 1, 6))[d]
            PathCollection_obj = ax.plot(
                [x[0] for x in xy_list], [y[1] for y in xy_list],
                alpha=0.75, label=patriarch_name, c=c
            )
            i -= 1
            d -= 1
        Legend_obj = ax.legend()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    def get_color_proximity(
        self, distance_df=None, color_dict=None, color_title='XKCD', color_str='Red'
    ):
        if color_dict is None: color_dict = mcolors.XKCD_COLORS
        if distance_df is None:
            colors_dict = {name: mcolors.to_rgb(color) for name, color in color_dict.items()}
            colors_df = self.colors_dict_to_dataframe(colors_dict)
            distance_df = self.get_distance_dataframe(colors_df, color_title=color_title)
        distance_dict = {}
        for name, color in color_dict.items():
            mask_series = (distance_df.index == name)
            column_name = f'distance_from_{color_str.lower()}'
            distance_list = distance_df[mask_series][column_name].tolist()
            if len(distance_list) == 1: distance_dict[name] = distance_list[0]
        color_tuple_list = sorted((distance, name) for name, distance in distance_dict.items())
        test_list = color_tuple_list[:32]
        
        return test_list
    
    def show_color_proximity(
        self, distance_df=None, color_dict=None, color_title='XKCD', color_str='Red',
        saeculum_title='Reformation', nearness_str='close to', verbose=True
    ):
        if color_dict is None: color_dict = mcolors.XKCD_COLORS
        if distance_df is None:
            colors_dict = {name: mcolors.to_rgb(color) for name, color in color_dict.items()}
            colors_df = self.colors_dict_to_dataframe(colors_dict)
            distance_df = self.get_distance_dataframe(colors_df, color_title=color_title)
        test_list = self.get_color_proximity(
            distance_df=distance_df, color_dict=color_dict, color_title=color_title, color_str=color_str
        )
        self.display_test_colors(
            test_list=test_list, saeculum_title=saeculum_title, face_title=color_str,
            nearness_str=nearness_str, color_dict=color_dict, color_title=color_title, face_point='Corner',
            verbose=verbose
        )
    
    def show_face_proximity(self, distance_df):
        for row_index, row_series in distance_df.iterrows():
            tuple_list = sorted(row_series.to_dict().items(), key=lambda x: x[1])
            if tuple_list[0][1] == 0.0:
                print('{} is in the {} corner'.format(row_index, tuple_list[0][0].split('_')[2]))
            else:
                corners_list = tuple_list[:3]
                face_set = set([corners_list[0][0].split('_')[2], corners_list[1][0].split('_')[2],
                                corners_list[2][0].split('_')[2]])
                if face_set in self.kryg_face_set_list:
                    print('{} is nearest the black-red-yellow-green face'.format(row_index))
                elif face_set in self.krmb_face_set_list:
                    print('{} is nearest the black-red-magenta-blue face'.format(row_index))
                elif face_set in self.kbcg_face_set_list:
                    print('{} is nearest the black-blue-cyan-green face'.format(row_index))
                elif face_set in self.wcgy_face_set_list:
                    print('{} is nearest the white-cyan-green-yellow face'.format(row_index))
                elif face_set in self.wcbm_face_set_list:
                    print('{} is nearest the white-cyan-blue-magenta face'.format(row_index))
                elif face_set in self.wyrm_face_set_list:
                    print('{} is nearest the white-yellow-red-magenta face'.format(row_index))
                else:
                    print('{} is nearest the {} face'.format(row_index, '-'.join(list(face_set))))
    
    def show_saeculum_image(self, saeculum_name):
        file_name = '{}.jpg'.format(saeculum_name)
        file_path = os.path.join(self.jpg_dir, file_name)
        jpg_image = Image.open(fp=file_path, mode='r')
        jpg_image = jpg_image.rotate(angle=180)
        width, height = jpg_image.size
        if (width > MAX_WIDTH) or (height > MAX_HEIGHT):
            if (width > MAX_WIDTH):
                multiple = MAX_WIDTH / width
                width *= multiple
                height *= multiple
            if (height > MAX_HEIGHT):
                multiple = MAX_HEIGHT / height
                width *= multiple
                height *= multiple
            jpg_image = jpg_image.resize(size=(int(width), int(height)), resample=0, box=None)
            width, height = jpg_image.size
        left = 0 - int(width/2)
        right = int(width/2)
        top = 0 - int(height/2)
        bottom = int(height/2)
        AxesImage_obj = imshow(X=np.asarray(jpg_image), origin='upper', extent=(left, right, bottom, top))
        
        return jpg_image
    
    def show_turning_image(self, year):
        mask_series = (self.turnings_df.turning_year_begin <= year) & (self.turnings_df.turning_year_end >= year)
        turning_name_list = self.turnings_df[mask_series].index.tolist()
        if len(turning_name_list):
            turning_name = turning_name_list[0]
            file_name = '{}.jpg'.format(turning_name)
            file_path = os.path.join(self.jpg_dir, file_name)
            jpg_image = Image.open(fp=file_path, mode='r')
            jpg_image = jpg_image.rotate(angle=180)
            width, height = jpg_image.size
            if (width > MAX_WIDTH) or (height > MAX_HEIGHT):
                if (width > MAX_WIDTH):
                    multiple = MAX_WIDTH / width
                    width *= multiple
                    height *= multiple
                if (height > MAX_HEIGHT):
                    multiple = MAX_HEIGHT / height
                    width *= multiple
                    height *= multiple
                jpg_image = jpg_image.resize(size=(int(width), int(height)), resample=0, box=None)
            width, height = jpg_image.size
        left = 0 - int(width/2)
        right = int(width/2)
        top = 0 - int(height/2)
        bottom = int(height/2)
        AxesImage_obj = imshow(X=np.asarray(jpg_image), origin='upper', extent=(left, right, bottom, top))
        
        return jpg_image
    
    def save_stopped_babbitt_plot_without_showing(
        self, history_theta_dict, stopped_year, history_year_dict,
        verbose=True, ax=None
    ):
        if ax is None:
            ax = plt.gca()
        
        # Turn interactive plotting off
        plt.ioff()
        
        # Create a new figure, plot into it, then close it so it never gets displayed
        fig = plt.figure(figsize=(13, 13))
        ax = fig.add_subplot(111, autoscale_on=False)
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        mask_series = (self.patriline_df.year_of_birth <= stopped_year)
        i = self.patriline_df[mask_series].shape[0]-1
        d = 5
        previous_saeculum = self.patriline_df[mask_series].head(1).saeculum_name.tolist()[0]
        for patriarch_name, row_series in self.patriline_df[mask_series].iterrows():
            start_year = int(row_series.year_of_birth)
            stop_year = row_series.year_of_death
            try:
                stop_year = int(stop_year)
            except:
                stop_year = start_year + 80
                if stop_year > max(history_year_dict.keys()):
                    stop_year = max(history_year_dict.keys())
            xy_list = self.get_one_stopped_arc(
                start_year=start_year, stop_year=stop_year,
                stopped_year=stopped_year, history_year_dict=history_year_dict,
                i=i
            )
            years_list = [start_year, stop_year]
            years_list = [year for year in years_list if year <= stopped_year]
            self.add_spiral_labels(years_list, history_year_dict, i)
            if stop_year > stopped_year:
                stop_year = stopped_year
            text_color = row_series.xkcd_text_color
            self.label_arc(
                start_year=start_year, stopped_year=stop_year,
                history_theta_dict=history_theta_dict,
                arc_label=patriarch_name,
                history_year_dict=history_year_dict, ideal_distance=13,
                i=i, label_color=text_color
            )
            saeculum = row_series.saeculum_name
            if saeculum != previous_saeculum:
                previous_saeculum = saeculum
                d = 5
            if verbose:
                print(f'patriarch_name, i, d, saeculum = "{patriarch_name, i, d, saeculum}"')
            cmap = self.saeculum_cmap_dict[saeculum]
            #c = plt.get_cmap(cmap)(np.linspace(0, 1, 6))[d]
            c = row_series.xkcd_color
            label_str = self.get_row_label(stopped_year, patriarch_name, row_series)
            PathCollection_obj = ax.plot(
                [xy[0] for xy in xy_list], [xy[1] for xy in xy_list],
                alpha=0.75, label=label_str, c=c, linewidth=9
            )
            i -= 1
            d -= 1
        
        #self.show_turning_image(stopped_year)
        legend_obj = ax.legend(ncol=2, loc='upper left')
        frame_obj = legend_obj.get_frame()
        frame_obj.set_facecolor('whitesmoke')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # Close it so it never gets displayed
        file_name = 'plot_{}.png'.format(stopped_year)
        file_path = os.path.join(self.bare_movie_folder, file_name)
        if verbose:
            print(f'Saving to {os.path.abspath(file_path)}')
        plt.savefig(file_path, format='png')
        plt.close(fig)
    
    def show_year_image(self, year):
        file_name = '{}.jpg'.format(year)
        file_path = os.path.join(self.jpg_dir, file_name)
        jpg_image = Image.open(fp=file_path, mode='r')
        jpg_image = jpg_image.rotate(angle=180)
        width, height = jpg_image.size
        if (width > MAX_WIDTH) or (height > MAX_HEIGHT):
            if (width > MAX_WIDTH):
                multiple = MAX_WIDTH / width
                width *= multiple
                height *= multiple
            if (height > MAX_HEIGHT):
                multiple = MAX_HEIGHT / height
                width *= multiple
                height *= multiple
            jpg_image = jpg_image.resize(
                size=(int(width), int(height)), resample=0, box=None
            )
            width, height = jpg_image.size
        left = 0 - int(width/2)
        right = int(width/2)
        top = 0 - int(height/2)
        bottom = int(height/2)
        AxesImage_obj = imshow(
            X=np.asarray(jpg_image), origin='upper',
            extent=(left, right, bottom, top)
        )
        
        return jpg_image
    
    def spirals(
        n_samples=100, noise=None, seed=None, mode='archimedes',
        n_loops=2, *args, **kwargs
    ):
        """
        Create spirals
        
        Currently only binary classification is supported for spiral generation
        
        Args:
            n_samples: int, number of datapoints to generate
            noise: float or None, standard deviation of the Gaussian noise added
            seed: int or None, seed for the noise
            n_loops: int, number of spiral loops, doesn't play well with 'bernoulli'
            mode: str, how the spiral should be generated. Current implementations:
                'archimedes': a spiral with equal distances between branches
                'bernoulli': logarithmic spiral with branch distances increasing
                'fermat': a spiral with branch distances decreasing (sqrt)
        
        Returns:
            Shuffled features and labels for 'spirals' synthetic dataset of type
            `base.Dataset`
        
        Raises:
            ValueError: If the generation `mode` is not valid
        
        TODO:
            - Generation of unbalanced data
        """
        
        n_classes = 2  # I am not sure how to make it multiclass
        
        _modes = {'archimedes': self.archimedes_spiral,
                  'bernoulli': self.bernoulli_spiral,
                  'fermat': self.fermat_spiral}
        
        if mode is None or mode not in _modes:
            raise ValueError('Cannot generate spiral with mode %s' % mode)
        
        if seed is not None:
            np.random.seed(seed)
        linspace = np.linspace(0, 2 * n_loops * np.pi, n_samples // n_classes)
        spir_x = np.empty(0, dtype=np.int32)
        spir_y = np.empty(0, dtype=np.int32)
        
        y = np.empty(0, dtype=np.int32)
        for label in range(n_classes):
            (base_cos, base_sin) = _modes[mode](
                linspace, label * np.pi, *args, **kwargs
            )
            spir_x = np.append(spir_x, base_cos)
            spir_y = np.append(spir_y, base_sin)
            y = np.append(
                y, label * np.ones(n_samples // n_classes, dtype=np.int32)
            )
        
        # Add more points if n_samples is not divisible by n_classes
        # (unbalanced!)
        extras = n_samples % n_classes
        if extras > 0:
            (x_extra, y_extra) = _modes[mode](
                np.random.rand(extras) * 2 * np.pi, *args, **kwargs
            )
            spir_x = np.append(spir_x, x_extra)
            spir_y = np.append(spir_y, y_extra)
            y = np.append(y, np.zeros(extras, dtype=np.int32))
        
        # Reshape the features/labels
        X = np.vstack((spir_x, spir_y)).T
        y = np.hstack(y)
        
        # Shuffle the data
        indices = np.random.permutation(range(n_samples))
        if noise is not None:
            X += np.random.normal(scale=noise, size=X.shape)
        
        return Dataset(data=X[indices], target=y[indices])
    
    def get_generation_name(self, year_of_birth):
        mask_series = (self.generations_df.birth_year_begin <= year_of_birth)
        mask_series = mask_series & (self.generations_df.birth_year_end >= year_of_birth)
        generation_name_list = self.generations_df[mask_series].index.tolist()
        if len(generation_name_list):
            generation_name = generation_name_list[0]
        else:
            generation_name = np.nan
        
        return generation_name
    
    def fit_year_curve(self, df=None, column_prefix='birth_year', ax=None):
        if ax is None:
            ax = plt.gca()
        if df is None:
            df = self.generations_df
        begin_column = '{}_begin'.format(column_prefix)
        end_column = '{}_end'.format(column_prefix)
        
        def func(x, a, b):
            
            return a*x + b
        
        columns_list = [begin_column, end_column]
        mask_series = False
        for column_name in columns_list:
            df[column_name] = pd.to_numeric(df[column_name])
            mask_series = mask_series | df[column_name].isnull()
        df = df[~mask_series][columns_list]
        begin_data = np.array(object=df[begin_column].tolist())
        end_data = np.array(object=df[end_column].tolist())
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(func, begin_data, end_data)
        line_2d_obj = ax.plot(begin_data, end_data, 'b-', label='data')

        def get_end_year(begin_year):
            '''Get the end year given the begin year'''

            return popt[0]*begin_year + popt[1]

        label_str = 'fit: end_year = %5.1f * begin_year + %5.1f' % tuple(popt)
        line_2d_obj = ax.plot(begin_data, get_end_year(begin_data), 'r-', label=label_str)
        ax.xlabel('Begin Year')
        ax.ylabel('End Year')
        legend_obj = ax.legend()
        
        return popt, get_end_year
    
    def get_event_story(self, event_year=None, event_name=None, verbose=False):
        if event_year is None: event_year = 1961
        if event_name is None: event_name = f'year of {event_year}'
        
        # Get patriline birth/death mask
        begin_mask_series = (self.patriline_df.year_of_birth <= event_year)
        # begin_mask_series |= self.patriline_df.year_of_birth.isnull()
        end_mask_series = (self.patriline_df.year_of_death >= event_year)
        end_mask_series |= self.patriline_df.year_of_death.isnull()
        mask_series = begin_mask_series & end_mask_series
        
        f_str = f'The members of the patriline that were alive during the {event_name} '
        f_str += 'were {}. '
        p_list = self.patriline_df[mask_series].index
        story_str = f_str.format(self.conjunctify_nouns(p_list)).replace('..', '.')
        f_str = '{} was {} years old at this time'
        patriarch_strs_list = []
        for patriarch, row_series in self.patriline_df[mask_series].iterrows():
            patriarch_str = f_str.format(patriarch, event_year - row_series.year_of_birth)
            patriarch_strs_list.append(patriarch_str)
        story_str += self.conjunctify_nouns(patriarch_strs_list)
        if not story_str.endswith('.'): story_str += '.'
        
        # Get Turning information
        column_descriptions_df = self.get_column_descriptions(self.turnings_df)
        mask_series = (column_descriptions_df.dtype != 'object')
        mask_series &= column_descriptions_df.column_name.map(lambda x: x.endswith('_begin') or x.endswith('_end'))
        df = column_descriptions_df[mask_series]
        mask_series = df.min_value.map(lambda x: event_year >= float(x)) & df.max_value.map(lambda x: event_year <= float(x))
        for column_name_prefix in set(df[mask_series].column_name.map(lambda x: x.replace('_begin', '').replace('_end', ''))):
            mask_series = (self.turnings_df[f'{column_name_prefix}_begin'] <= event_year)
            mask_series &= (self.turnings_df[f'{column_name_prefix}_end'] >= event_year)
            if mask_series.any():
                turning_name = self.turnings_df[mask_series].iloc[-1].name
                saeculum_name = self.turnings_df[mask_series].iloc[-1].saeculum_name
                entering_elderhood = self.turnings_df[mask_series].iloc[-1].entering_elderhood
                # if verbose: print(entering_elderhood); display(self.turnings_df[mask_series])
                if not entering_elderhood.endswith('s'): entering_elderhood += ' Generation'
                entering_midlife = self.turnings_df[mask_series].iloc[-1].entering_midlife
                if not entering_midlife.endswith('s'): entering_midlife += ' Generation'
                entering_young_adulthood = self.turnings_df[mask_series].iloc[-1].entering_young_adulthood
                if not entering_young_adulthood.endswith('s'): entering_young_adulthood += ' Generation'
                entering_childhood = self.turnings_df[mask_series].iloc[-1].entering_childhood
                if not entering_childhood.endswith('s'): entering_childhood += ' Generation'
                if not story_str.endswith(' '): story_str += ' '
                story_str += f'This year was during the {turning_name}, a turning of the {saeculum_name} saeculum.'
                story_str += f' During this era the {entering_elderhood} were entering elderhood, the {entering_midlife} were'
                story_str += f' entering midlife, the {entering_young_adulthood} were entering young adulthood, and the'
                story_str += f' {entering_childhood} were entering childhood.'
        
        # Get saecular Awakening/Crisis information
        column_descriptions_df = self.get_column_descriptions(self.saecula_df)
        mask_series = (column_descriptions_df.dtype != 'object')
        mask_series &= column_descriptions_df.column_name.map(lambda x: x.endswith('_year_begin') or x.endswith('_year_end'))
        df = column_descriptions_df[mask_series]
        mask_series = df.min_value.map(lambda x: event_year >= float(x)) & df.max_value.map(lambda x: event_year <= float(x))
        prefixes_list = df[mask_series].column_name.map(lambda x: x.replace('_year_begin', '').replace('_year_end', ''))
        for column_name_prefix in set(prefixes_list):
            begin_mask_series = (self.saecula_df[f'{column_name_prefix}_year_begin'] <= event_year)
            end_mask_series = (self.saecula_df[f'{column_name_prefix}_year_end'] >= event_year)
            mask_series = begin_mask_series & end_mask_series
            if mask_series.any():
                saeculum_name = self.saecula_df[mask_series].iloc[-1].name
                solstice_name = self.saecula_df[mask_series].iloc[-1][f'{column_name_prefix}_name']
                solstice_name = f"{saeculum_name} saeculum's {solstice_name}"
                climax_mask_series = (self.saecula_df[f'{column_name_prefix}_climax_year'] == event_year)
                if self.saecula_df[climax_mask_series].shape[0]: story_str += f' The {solstice_name} was climaxing at this time.'
        
        return story_str
    
    def get_column_descriptions(self, df, column_list=None):
        if column_list is None: column_list = df.columns
        groups_dict = df.columns.to_series().groupby(df.dtypes).groups
        rows_list = []
        for dtype, dtype_column_list in groups_dict.items():
            for column_name in dtype_column_list:
                if column_name in column_list:
                    null_mask_series = df[column_name].isnull()
                    blank_mask_series = df[column_name].map(lambda x: not len(str(x)))
                    mask_series = null_mask_series | blank_mask_series

                    # Get input row in dictionary format; key = col_name
                    row_dict = {}
                    row_dict['column_name'] = column_name
                    row_dict['dtype'] = str(dtype)
                    row_dict['count_nulls'] = null_mask_series.sum()
                    row_dict['count_blanks'] = blank_mask_series.sum()

                    # Count how many unique numbers there are
                    try:
                        row_dict['count_uniques'] = df[column_name].nunique()
                    except Exception:
                        row_dict['count_uniques'] = np.nan

                    # Count how many zeroes the column has
                    try:
                        row_dict['count_zeroes'] = int((df[column_name] == 0).sum())
                    except Exception:
                        row_dict['count_zeroes'] = np.nan

                    # Check to see if the column has any dates
                    date_series = pd.to_datetime(df[column_name], errors='coerce')
                    null_series = date_series[~date_series.notnull()]
                    row_dict['has_dates'] = (null_series.shape[0] < date_series.shape[0])

                    # Show the minimum value in the column
                    try:
                        row_dict['min_value'] = df[~mask_series][column_name].min()
                    except Exception:
                        row_dict['min_value'] = np.nan

                    # Show the maximum value in the column
                    try:
                        row_dict['max_value'] = df[~mask_series][column_name].max()
                    except Exception:
                        row_dict['max_value'] = np.nan

                    # Show whether the column contains only integers
                    try:
                        oib = (df[column_name].apply(lambda x: float(x).is_integer())).all()
                        row_dict['only_integers'] = oib
                    except Exception:
                        row_dict['only_integers'] = float('nan')

                    rows_list.append(row_dict)

        columns_list = ['column_name', 'dtype', 'count_nulls', 'count_blanks',
                        'count_uniques', 'count_zeroes', 'has_dates',
                        'min_value', 'max_value', 'only_integers']
        blank_ranking_df = pd.DataFrame(rows_list, columns=columns_list)

        return(blank_ranking_df)
    
    def get_max_rsquared_adj(self, df, columns_list, verbose=False):
        if verbose:
            t0 = time.time()
        rows_list = []
        n = len(columns_list)
        import statsmodels.api as sm
        for i in range(n-1):
            first_column = columns_list[i]
            first_series = df[first_column]
            max_correlation = 0.0
            max_column = first_column
            for j in range(i+1, n):
                second_column = columns_list[j]
                second_series = df[second_column]
                
                # Assume the first column is never identical to the second column
                X, y = first_series.values.reshape(-1, 1), second_series.values.reshape(-1, 1)
                
                # Compute with statsmodels, by adding intercept manually
                X1 = sm.add_constant(X)
                result = sm.OLS(y, X1).fit()
                this_correlation = abs(result.rsquared_adj)
                
                if this_correlation > max_correlation:
                    max_correlation = this_correlation
                    max_column = second_column

            # Get input row in dictionary format; key = col_name
            row_dict = {}
            row_dict['reference_column'] = first_column
            row_dict['max_column'] = max_column
            row_dict['columns_correlation'] = max_correlation

            rows_list.append(row_dict)

        column_list = ['reference_column', 'max_column', 'columns_correlation']
        column_similarities_df = pd.DataFrame(rows_list, columns=column_list)
        if verbose:
            t1 = time.time()
            print(t1-t0, time.ctime(t1))

        return column_similarities_df
    
    def show_3d_plot(
        self, three_d_df, z_column='Red', x_column='Green',
        y_column='Blue', pane_color='white', ax=None, verbose=False
    ):
        
        # Create the figure and subplot
        if ax is None: ax = plt.figure(figsize=(18, 8)).add_subplot(projection='3d')

        # Change the 2 walls of the 3d plot to black
        if verbose: print([f'ax.xaxis.{fn}' for fn in dir(ax.xaxis) if 'color' in fn])
        ax.set_axisbelow(True)
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)
        
        xlabel_text = ax.set_xlabel(x_column)
        ylabel_text = ax.set_ylabel(y_column)
        zlabel_text = ax.set_zlabel(z_column)
        columns_list = [x_column, y_column, z_column]
        df = three_d_df[columns_list].dropna(axis='index', how='any')
        pca_ndarray = df.values
        path_collection = ax.scatter(pca_ndarray[:, 0], pca_ndarray[:, 1],
                                     pca_ndarray[:, 2], alpha=0.75, c=df.index)
        title_text = 'Scatterplot of the {}, {}, and {} Data'
        text_obj = ax.set_title(title_text.format(x_column, y_column, z_column))
    
    def get_begin_turning_name(self, year):
        begin_turning_name = None
        begin_mask_series = (self.turnings_df.turning_year_begin == year)
        if self.turnings_df[begin_mask_series].shape[0]:
            begin_turning_name = self.turnings_df[begin_mask_series].index.array[0]
            if begin_turning_name.lower().startswith('the '):
                begin_turning_name = begin_turning_name[4:]
        
        return begin_turning_name
    
    def get_facts_about(self, year):
        
        # Get generational information
        mask_series = (self.generations_df.birth_year_begin <= year)
        mask_series &= (self.generations_df.birth_year_end >= year)
        df = self.generations_df[mask_series]
        if df.shape[0]:
            generation_name = df.index.array[0]
            generations_archetype = self.generations_df[mask_series].generations_archetype.squeeze().lower()
            print(f'The {generations_archetype} {generation_name} generation were being born at this time.')
        
        # Get Babbitt information
        mask_series = (self.patriline_df.year_of_birth <= year)
        end_mask_series = (self.patriline_df.year_of_death >= year)
        mask_series &= (end_mask_series | self.patriline_df.year_of_death.isnull())
        patriarch_names_list = self.patriline_df[mask_series].sort_values('year_of_birth').index.tolist()
        if patriarch_names_list:
            waswere = 'were' if len(patriarch_names_list) > 1 else 'was'
            print(self.conjunctify_nouns(patriarch_names_list) + f' {waswere} alive during this year.')
        
        # Get saecular Awakening information
        mask_series = (self.saecula_df.awakening_year_begin <= year)
        mask_series &= (self.saecula_df.awakening_year_end >= year)
        if mask_series.any():
            saeculum_name = self.saecula_df[mask_series].index.array[0]
            climax_mask_series = (self.saecula_df.awakening_climax_year == year)
            if self.saecula_df[climax_mask_series].shape[0]:
                awakening_name = self.saecula_df[climax_mask_series].awakening_name.squeeze()
                awakening_name = f"{saeculum_name} saeculum's {awakening_name}"
                print(f'The {awakening_name} was climaxing at this time.')
            else:
                awakening_name = f"{saeculum_name} saeculum's {self.saecula_df[mask_series].awakening_name.squeeze()}"
                print(f'This year was during the {awakening_name}.')
        
        # Get saecular Crisis information
        mask_series = (self.saecula_df.crisis_year_begin <= year) & (self.saecula_df.crisis_year_end >= year)
        if mask_series.any():
            saeculum_name = self.saecula_df[mask_series].index.array[0]
            climax_mask_series = (self.saecula_df.crisis_climax_year == year)
            if self.saecula_df[climax_mask_series].shape[0]:
                crisis_name = f"{saeculum_name} saeculum's {self.saecula_df[climax_mask_series].crisis_name.squeeze()}"
                print(f'The {crisis_name} was climaxing at this time.')
            else:
                crisis_name = f"{saeculum_name} saeculum's {self.saecula_df[mask_series].crisis_name.squeeze()}"
                print(f'This year was during the {crisis_name}.')
        
        # Get Turning information
        end_mask_series = (self.turnings_df.turning_year_end == year)
        if self.turnings_df[end_mask_series].shape[0]:
            end_turning_name = self.turnings_df[end_mask_series].index.array[0]
            if end_turning_name.lower().startswith('the '):
                end_turning_name = end_turning_name[4:]
            print(f'This year was the end of the {end_turning_name}', end='')
            begin_turning_name = self.get_begin_turning_name(year)
            if begin_turning_name is not None:
                print(f', and the beginning of the {begin_turning_name}.')
            else: print('.')
        else:
            begin_turning_name = self.get_begin_turning_name(year)
            if begin_turning_name is not None:
                print(f'This year was the beginning of the {begin_turning_name}.')
        
        # Get Turning notes
        mask_series = (self.turnings_df.turning_year_end > year) & (self.turnings_df.turning_year_begin <= year)
        if mask_series.any():
            turning_notes = self.turnings_df[mask_series].turning_notes.squeeze().strip()
            print(f'turning_notes = "{turning_notes}"')
        
        # Get presidential information
        mask_series = (self.us_presidents_df.year_reign_end >= year)
        mask_series &= (self.us_presidents_df.year_reign_begin <= year)
        if mask_series.any():
            president_name = self.us_presidents_df[mask_series].index.array[0]
            print(f'{president_name} was president during this year.')
        
        # Get Billboard Year-End number-one singles information
        self.show_billboard_year_end_number_one_single(year)
    
    def show_number_one_single(self, nearest_issue_date):
        if self.s.pickle_exists('billboard_df'):
            df = self.s.load_object('billboard_df')
            if 'issue_date' in df.columns:
                row_series = sorted([rs for (i, rs) in df.iterrows()], key=lambda x: abs((x.issue_date - nearest_issue_date).days))[0]
                song_name = row_series.song_name
                artist_name = row_series.artist_name
                print(f'The number-one single for this date was "{song_name}" by {artist_name}.')
        else:
            self.show_billboard_year_end_number_one_single(nearest_issue_date.year)
    
    def show_billboard_year_end_number_one_single(self, year):
        if self.s.pickle_exists('Billboard_Year_End_number_one_singles_df'):
            df = self.s.load_object('Billboard_Year_End_number_one_singles_df')
            if 'Year' in df.columns:
                mask_series = (df.Year == year)
                if df[mask_series].shape[0] and ('song_name' in df.columns) and ('artist_name' in df.columns):
                    song_name = df[mask_series].song_name.squeeze()
                    artist_name = df[mask_series].artist_name.squeeze()
                    print(f'The Billboard Year-End number-one single for that year was "{song_name}" by {artist_name}.')
    
    #################### Elliptical Functions ####################
    
    def elliptical_polar_to_cartesian(self, theta, vertical_radius, horizontal_radius=None, aspect_ratio=1.1193862644404413):
        """
        From https://math.stackexchange.com/questions/315386/ellipse-in-polar-coordinates
        and https://mathworld.wolfram.com/Ellipse.html
        """
        if horizontal_radius is None:
            horizontal_radius = vertical_radius * aspect_ratio
        radians = theta*(pi/180)
        x = sqrt(vertical_radius**2 - (vertical_radius*cos(radians))**2)*horizontal_radius/vertical_radius
        y = sqrt(horizontal_radius**2 - (horizontal_radius*sin(radians))**2)*vertical_radius/horizontal_radius
        
        theta_prime = theta % 360
        if theta_prime >= 270:
            x = -x
            y = -y
        elif theta_prime >= 180:
            x = -x
            y = y
        elif theta_prime >= 90:
            x = x
            y = y
        else:
            x = x
            y = -y
        
        return int(x), int(y)
    
    def add_elliptical_spiral_labels(
        self, years_list, history_year_dict, i=0,
        aspect_ratio=1.1193862644404413, ax=None):
        if ax is None:
            ax = plt.gca()
        i = i % 4
        for year in years_list:
            vertical_radius, theta = history_year_dict[year]
            vertical_radius += 25*i
            vertical_radius -= 25/2
            horizontal_radius = vertical_radius * aspect_ratio
            x, y = self.elliptical_polar_to_cartesian(
                theta=theta, vertical_radius=vertical_radius,
                horizontal_radius=horizontal_radius, aspect_ratio=aspect_ratio
            )
            text_obj = ax.text(x, y, year, fontsize=10, color='gray',
                                rotation=theta-90, rotation_mode='anchor')
    
    def create_elliptical_xy_list(
        self, history_radius_dict, aspect_ratio=1.1193862644404413
    ):
        xy_list = []
        for vertical_radius in sorted(history_radius_dict.keys()):
            year, theta = history_radius_dict[vertical_radius]
            horizontal_radius = vertical_radius * aspect_ratio
            cartesian_tuple = self.elliptical_polar_to_cartesian(
                theta=theta, vertical_radius=vertical_radius,
                horizontal_radius=horizontal_radius, aspect_ratio=aspect_ratio
            )
            if len(xy_list):
                if (cartesian_tuple != xy_list[-1]):
                    xy_list.append(cartesian_tuple)
            else:
                xy_list.append(cartesian_tuple)
        
        return xy_list
    
    def get_one_elliptical_arc(
        self, start_year, stop_year, history_year_dict,
        theta_offset=0, radius_offset=0,
        i=0, aspect_ratio=1.1193862644404413, verbose=False
    ):
        i = i % 4
        if verbose:
            if start_year in history_year_dict:
                print(f'history_year_dict[start_year] = history_year_dict[{start_year}] = {history_year_dict[start_year]}')
        start_radius = history_year_dict[start_year][0]
        start_radius += 25*i
        stop_radius = history_year_dict[stop_year][0]
        stop_radius += 25*i
        increment_count = int(2*pi*start_radius)
        radius_array = np.linspace(start=start_radius, stop=stop_radius,
                                   num=increment_count)
        start_theta = history_year_dict[start_year][1]
        if verbose:
            print(f'start_theta = {start_theta}')
        stop_theta = history_year_dict[stop_year][1]
        if verbose:
            print(f'stop_theta = {stop_theta}')
        theta_array = np.linspace(start=start_theta, stop=stop_theta,
                                  num=increment_count)
        xy_list = []
        for vertical_radius, theta in zip(radius_array, theta_array):
            vertical_radius += radius_offset
            horizontal_radius = vertical_radius * aspect_ratio
            theta += theta_offset
            cartesian_tuple = self.elliptical_polar_to_cartesian(
                theta=theta, vertical_radius=vertical_radius,
                horizontal_radius=horizontal_radius, aspect_ratio=aspect_ratio
            )
            if len(xy_list):
                if (cartesian_tuple != xy_list[-1]):
                    xy_list.append(cartesian_tuple)
            else:
                xy_list.append(cartesian_tuple)
        
        return xy_list
    
    def label_ellipse(
        self, start_year, stopped_year, history_theta_dict, elliptical_label,
        history_year_dict, ideal_distance=13, i=0,
        label_color='black', aspect_ratio=1.1193862644404413, ax=None
    ):
        if ax is None:
            ax = plt.gca()
        i = i % 4
        starting_year = int(((start_year + stopped_year) / 2) - (len(elliptical_label) / 2))
        starting_radius, starting_theta = history_year_dict[starting_year]
        next_radius, next_theta = history_year_dict[starting_year+1]
        
        # Tan(A) = Opposite/Adjacent
        radians = atan(ideal_distance/starting_radius)
        theta_sign = np.sign(next_theta-starting_theta)
        
        # Increment the theta so that it spaces the letters the same regardless of the vertical radius
        theta_increment = theta_sign*(radians*180/pi)
        
        # Figure out if you have to flip the characters upside-down and place them in right-to-left order
        if (starting_theta%360) < 200:
            #logging.info('')
            #logging.info('Right-side up thetas:')
            theta = starting_theta
            vertical_radius = starting_radius + 25*i
            vertical_radius -= 25/2
            for c in elliptical_label[::int(-theta_sign)]:
                #logging.info('c: "{}", vertical_radius: "{}", theta: "{}"'.format(c, vertical_radius, theta % 360))
                horizontal_radius = vertical_radius * aspect_ratio
                x, y = self.elliptical_polar_to_cartesian(
                    theta=theta, vertical_radius=vertical_radius,
                    horizontal_radius=horizontal_radius,
                    aspect_ratio=aspect_ratio
                )
                text_obj = ax.text(x, y, c, fontsize=12, color=label_color,
                                    rotation=theta-90, rotation_mode='anchor')
                theta += theta_increment
                if int(theta) in history_theta_dict:
                    vertical_radius = history_theta_dict[int(theta)][1]
                    vertical_radius += 25*i
                    vertical_radius -= 25/2
        else:
            logging.info('')
            logging.info('Upside-down thetas:')
            theta = starting_theta + theta_increment*len(elliptical_label)
            if int(theta) in history_theta_dict:
                vertical_radius = history_theta_dict[int(theta)][1]
                vertical_radius += 25*i
            else:
                vertical_radius = starting_radius + 25*i
            vertical_radius += 25/2
            for c in elliptical_label[::int(-theta_sign)]:
                logging.info('c: "{}", vertical_radius: "{}", theta: "{}"'.format(c, vertical_radius, theta % 360))
                horizontal_radius = vertical_radius * aspect_ratio
                x, y = self.elliptical_polar_to_cartesian(
                    theta=theta, vertical_radius=vertical_radius,
                    horizontal_radius=horizontal_radius,
                    aspect_ratio=aspect_ratio
                )
                text_obj = ax.text(x, y, c, fontsize=12, color=label_color,
                                    rotation=theta+90, rotation_mode='anchor')
                theta -= theta_increment
                if int(theta) in history_theta_dict:
                    vertical_radius = history_theta_dict[int(theta)][1]
                    vertical_radius += 25*i
                    vertical_radius += 25/2
    
    def show_elliptical_babbitt_plot(
        self, history_theta_dict, history_year_dict,
        aspect_ratio=1.1193862644404413, ax=None, verbose=False
    ):
        if ax is None:
            ax = plt.gca()
        ax.set_xlim(-1000*aspect_ratio, 1000*aspect_ratio)
        ax.set_ylim(-1000, 1000)
        axis_tuple = ax.axis('equal')
        i = self.patriline_df.shape[0]-1
        d = 5
        previous_saeculum = self.patriline_df.head(1).saeculum_name.tolist()[0]
        for patriarch_name, row_series in self.patriline_df.iterrows():
            start_year = int(row_series.year_of_birth)
            stop_year = row_series.year_of_death
            try:
                stop_year = int(stop_year)
            except:
                stop_year = start_year + 80
                if stop_year > max(history_year_dict.keys()):
                    stop_year = max(history_year_dict.keys())
            xy_list = self.get_one_elliptical_arc(
                start_year=start_year, stop_year=stop_year,
                history_year_dict=history_year_dict, i=i,
                aspect_ratio=aspect_ratio
            )
            self.add_elliptical_spiral_labels(
                [start_year, stop_year], history_year_dict, i,
                aspect_ratio=aspect_ratio
            )
            self.label_ellipse(
                start_year=start_year, stopped_year=stop_year,
                history_theta_dict=history_theta_dict,
                elliptical_label=patriarch_name,
                history_year_dict=history_year_dict, ideal_distance=13,
                i=i, label_color='black', aspect_ratio=aspect_ratio
            )
            saeculum = row_series.saeculum_name
            if saeculum != previous_saeculum:
                previous_saeculum = saeculum
                d = 5
            if verbose:
                print(f'patriarch_name, i, d, saeculum = "{patriarch_name, i, d, saeculum}"')
            cmap = self.saeculum_cmap_dict[saeculum]
            c = plt.get_cmap(cmap)(np.linspace(0, 1, 6))[d]
            x_list = [x[0] for x in xy_list]
            y_list = [y[1] for y in xy_list]
            PathCollection_obj = ax.plot(
                x_list, y_list, alpha=0.75, label=patriarch_name, c=c
            )
            i -= 1
            d -= 1
        Legend_obj = ax.legend()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    def plot_elliptical_year(
        self, history_year_dict, bottom_year,
        aspect_ratio=1.1193862644404413, radius_offset=None,
        ax=None, verbose=False
    ):
        if ax is None:
            ax = plt.gca()
        theta_offset = 0
        if (bottom_year in history_year_dict.keys()):
            theta_offset = 0 - history_year_dict[bottom_year][1]
        
        # Get the radius offset
        max_year = max(history_year_dict.keys())
        max_radius = history_year_dict[max_year][0]
        self.set_invisible_ellipse(vertical_radius=max_radius, ax=ax)
        if radius_offset is None:
            radius_offset = max_radius - history_year_dict[bottom_year][0]
        
        # Get living patriachs dataset
        birth_series = self.patriline_df.year_of_birth
        death_series = self.patriline_df.year_of_death
        mask_series = (bottom_year >= birth_series)
        patriarch_names_list = self.patriline_df[mask_series].index.tolist()
        mask_series &= (bottom_year <= death_series) | death_series.isnull()
        labels_list = self.patriline_df[mask_series].index.tolist()
        
        # Add theta- and radius-adjusted patriarch arcs
        for i, patriarch_name in enumerate(patriarch_names_list):
            self.plot_elliptical_patriarch(
                patriarch_name, history_year_dict, bottom_year,
                aspect_ratio=aspect_ratio,
                theta_offset=theta_offset, radius_offset=radius_offset,
                i=i, ax=ax,
                add_label=bool(patriarch_name in labels_list)
            )
        
        # Label and adjust plot
        text_obj = ax.text(
            0.5, 0.0625, bottom_year, fontsize=12, color='black',
            ha='center', transform=ax.transAxes
        )
        self.adjust_axis(ax=ax)
    
    def plot_elliptical_patriarch(
        self, patriarch_name, history_year_dict, bottom_year,
        aspect_ratio=1.1193862644404413,
        theta_offset=0, radius_offset=0, i=0, ax=None, add_label=True
    ):
        if ax is None:
            ax = plt.gca()
        
        mask_series = (self.patriline_df.index == patriarch_name)
        start_year = int(self.patriline_df[mask_series].year_of_birth.tolist()[0])
        stop_year = self.patriline_df[mask_series].year_of_death.tolist()[0]
        try:
            stop_year = int(stop_year)
        except:
            stop_year = max(history_year_dict.keys())
        xy_list = self.get_one_elliptical_arc(
            start_year=start_year, stop_year=min(stop_year, bottom_year),
            history_year_dict=history_year_dict,
            theta_offset=theta_offset, radius_offset=radius_offset, i=i,
            aspect_ratio=aspect_ratio
        )
        css4_color = self.patriline_df[mask_series].css4_color.squeeze()
        PathCollection_obj = ax.plot(
            [x[0] for x in xy_list], [y[1] for y in xy_list],
            color=css4_color, alpha=0.5
        )
        if add_label:
            self.add_elliptical_patriarch_label(
                patriarch_name=patriarch_name,
                history_year_dict=history_year_dict, bottom_year=bottom_year,
                aspect_ratio=aspect_ratio,
                theta_offset=theta_offset, radius_offset=radius_offset, i=i
            )
    
    def set_invisible_ellipse(
        self, vertical_radius=700, aspect_ratio=1.1193862644404413,
        ax=None, verbose=False
    ):
        if ax is None:
            ax = plt.gca()
        x_list = []
        y_list = []
        for theta in range(360):
            horizontal_radius = vertical_radius * aspect_ratio
            x, y = self.elliptical_polar_to_cartesian(
                theta=theta, vertical_radius=vertical_radius,
                horizontal_radius=horizontal_radius, aspect_ratio=aspect_ratio
            )
            x_list.append(x)
            y_list.append(y)
        PathCollection_obj = ax.plot(x_list, y_list, color='w', alpha=0.0)
    
    def add_elliptical_theta_labels(
        self, starting_theta=270, years_list=None, history_year_dict=None,
        bottom_year=None, aspect_ratio=1.1193862644404413, ax=None,
        verbose=False
    ):
        if ax is None:
            ax = plt.gca()
        x_list = []
        y_list = []
        def append_lists(theta, vertical_radius=128):
            horizontal_radius = vertical_radius * aspect_ratio
            x, y = self.elliptical_polar_to_cartesian(
                theta=theta, vertical_radius=vertical_radius,
                horizontal_radius=horizontal_radius, aspect_ratio=aspect_ratio
            )
            x_list.append(x)
            y_list.append(y)
            thetas_list.append(theta)
        if history_year_dict is None:
            addend = 360//64
            theta = 0 - addend
            thetas_list = []
            while theta < (360 - addend):
                theta += addend
                append_lists(theta)
            labels_list = thetas_list
        else:
            thetas_list = []
            labels_list = []
            theta_offset = 0
            if (bottom_year is not None):
                if (bottom_year in history_year_dict.keys()):
                    theta_offset = starting_theta - history_year_dict[bottom_year][1]
            if years_list is None:
                for year, (_, theta) in history_year_dict.items():
                    if isinstance(theta, (int, float)):
                        append_lists(theta+theta_offset)
                        labels_list.append(year)
            else:
                for year in years_list:
                    _, theta = history_year_dict[year]
                    if isinstance(theta, (int, float)):
                        append_lists(theta+theta_offset)
                        labels_list.append(year)
        for x, y, theta, label in zip(
            self.min_max_norm(x_list),
            self.min_max_norm(y_list),
            thetas_list,
            labels_list
        ):
            if verbose:
                print(x, y, theta, label)
            if (theta % 90):
                text_obj = ax.text(
                    x, y, label, fontsize=10, color='gray', ha='center',
                    va='center', transform=ax.transAxes
                )
            else:
                text_obj = ax.text(
                    x, y, label, fontsize=10, color='green', ha='center',
                    va='center', transform=ax.transAxes, weight='bold'
                )
        self.adjust_axis(ax=ax)
    
    def add_elliptical_patriarch_label(
        self, patriarch_name, history_year_dict, bottom_year,
        aspect_ratio=1.1193862644404413,
        theta_offset=0, radius_offset=0, i=0, ax=None
    ):
        if ax is None:
            ax = plt.gca()
        i = i % 4
        vertical_radius, theta = history_year_dict[bottom_year]
        theta += theta_offset
        vertical_radius += 25*i
        vertical_radius -= 25/2
        vertical_radius += radius_offset
        horizontal_radius = vertical_radius * aspect_ratio
        x, y = self.elliptical_polar_to_cartesian(
            theta=theta, vertical_radius=vertical_radius,
            horizontal_radius=horizontal_radius, aspect_ratio=aspect_ratio
        )
        text_obj = ax.text(
            x, y, patriarch_name, fontsize=8, color='gray',
            ha='center', va='center'
        )