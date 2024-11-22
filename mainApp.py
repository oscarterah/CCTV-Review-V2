# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 02:39:18 2024

@author: oscar
"""

from plyer.facades import orientation
from kivymd.app import MDApp
from VideoApp import MainWid
from kivy.uix.screenmanager import ScreenManager, NoTransition
from LiveStreamApp import LiveViewScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.screen import Screen
from kivymd.uix.screen import MDScreen
from kivymd.uix.navigationrail import MDNavigationRail
from AnalyticsApp import AnalyticsScreen
from SettingsApp import SettingsScreen



class MyRailScreen(MDNavigationRail):
    def __init__(self, mainwid, **kwargs):
        super().__init__()
        self.mainwid = mainwid


    def goto_videoapp(self):
        self.mainwid.myscreenmanager.goto_videoapp()

    def goto_liveapp(self):
        self.mainwid.myscreenmanager.goto_liveapp()
        
    def goto_analyticsapp(self):
        self.mainwid.myscreenmanager.goto_analyticsapp()
        
    def goto_settingsapp(self):
        self.mainwid.myscreenmanager.goto_settingsapp()


class BaseScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__()
        self.myrailscreen = MyRailScreen(self)
        self.main_layout = MDBoxLayout(orientation="horizontal")

        self.myscreenmanager = MyScreenManager()

        self.main_layout.add_widget(self.myrailscreen)
        self.main_layout.add_widget(self.myscreenmanager)

        self.add_widget(self.main_layout)
        
        

class MyScreenManager(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__()
        self.transition = NoTransition()
        self.wid_videoapp = Screen(name="VideoApp")
        self.wid_liveapp = Screen(name="LiveApp")
        self.wid_analytics = Screen(name="AnalyticsApp")
        self.wid_settings = Screen(name="SettingsApp")


        mainwid_screen = MainWid(self)
        livestream_screen = LiveViewScreen(self)
        analytics_screen = AnalyticsScreen(self)
        settings_screen = SettingsScreen(self)

        self.wid_videoapp.add_widget(mainwid_screen)
        self.wid_liveapp.add_widget(livestream_screen)
        self.wid_analytics.add_widget(analytics_screen)
        self.wid_settings.add_widget(settings_screen)
        
        self.add_widget(self.wid_videoapp)
        self.add_widget(self.wid_liveapp)
        self.add_widget(self.wid_analytics)
        self.add_widget(self.wid_settings)
        # self.goto_videoapp()
        self.goto_liveapp()

    
    def goto_videoapp(self, *args):
        self.current = "VideoApp"

    def goto_liveapp(self, *args):
        self.current = "LiveApp"
        
    def goto_analyticsapp(self, *args):
        self.current = "AnalyticsApp"

    def goto_settingsapp(self, *args):
        self.current = "SettingsApp"


class MainApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme_cls.theme_style = "Dark"

    def build(self):
        return BaseScreen()


if __name__ == '__main__':
    MainApp().run()
