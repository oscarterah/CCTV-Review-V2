# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 02:39:18 2024

@author: oscar
"""

from kivymd.app import MDApp
from VideoApp import MainWid
from kivymd.uix.screenmanager import MDScreenManager
from LiveStreamApp import LiveViewScreen
from kivymd.uix.screen import Screen


class ScreenManager(MDScreenManager):
    def __init__(self, **kwargs):
        super().__init__()
        self.mainwid_screen = MainWid()
        self.livestream_screen = LiveViewScreen()
        
        
        
        wid = Screen(name="VideoApp")
        wid.add_widget(self.mainwid_screen)
        self.add_widget(wid)
        
        
        
        
        self.add_widget(self.mainwid_screen)
        self.add_widget(self.livestream_screen)
        



















class MainApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme_cls.theme_style = "Dark"

    def build(self):
        return ScreenManager()


if __name__ == '__main__':
    MainApp().run()
