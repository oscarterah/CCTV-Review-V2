# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 02:39:18 2024

@author: oscar
"""

from kivymd.app import MDApp
from VideoApp import MainWid
from kivy.uix.screenmanager import ScreenManager, NoTransition
from LiveStreamApp import LiveViewScreen
from kivymd.uix.screen import Screen


class MyScreenManager(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__()
        self.transition = NoTransition()
        self.mainwid_screen = MainWid(self)
        self.livestream_screen = LiveViewScreen(self)
        
        
        
        wid_videoapp = Screen(name="VideoApp")
        wid_videoapp.add_widget(self.mainwid_screen)
        self.add_widget(wid_videoapp)
        
        wid_liveapp = Screen(name="LiveApp")
        wid_liveapp.add_widget(self.livestream_screen)
        self.add_widget(wid_liveapp)
        
        self.goto_videoapp()
    
    def goto_videoapp(self, *args):
        self.current = "VideoApp"

    def goto_liveapp(self, *args):
        self.current = "LiveApp"



class MainApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme_cls.theme_style = "Dark"

    def build(self):
        return MyScreenManager()


if __name__ == '__main__':
    MainApp().run()
