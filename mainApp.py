# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 02:39:18 2024

@author: oscar
"""

from kivymd.app import MDApp
from VideoApp import MainWid
from kivymd.uix.screenmanager import MDScreenManager




class ScreenManager(MDScreenManager):
    def __init__(self, **kwargs):
        super().__init__()
        self.mainwid_screen = MainWid()
        self.add_widget(self.mainwid_screen)
        



















class MainApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme_cls.theme_style = "Dark"

    def build(self):
        return ScreenManager()


if __name__ == '__main__':
    MainApp().run()