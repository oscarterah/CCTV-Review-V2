# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:59:42 2024

@author: oscar
"""

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.screen import MDScreen
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.label import MDLabel

class SettingsScreen(MDScreen):
    def __init__(self, mainwid, **kwargs):
        super().__init__(**kwargs)
        self.mainwid = mainwid 
        # Create layout
        layout = MDBoxLayout(orientation='vertical')
        
        # Add toolbar
        toolbar = MDTopAppBar(
            title="Settings",
            right_action_items=[
                ["view-grid", lambda x: self.toggle_view("grid")],
                ["view-list", lambda x: self.toggle_view("list")]
            ]
        )
        
        # Add content
        content = MDLabel(
            text="Live Camera Feeds Will Appear Here",
            halign='center'
        )
        
        layout.add_widget(toolbar)
        layout.add_widget(content)
        self.add_widget(layout)
    
    def videoapp(self):
        self.mainwid.goto_videoapp()

    def liveapp(self):
        self.mainwid.goto_liveapp()


    def toggle_view(self, view_type):
        print(f"Switching to {view_type} view")