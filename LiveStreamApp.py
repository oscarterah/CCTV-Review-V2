# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 03:04:38 2024

@author: oscar
"""
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.screen import MDScreen
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.label import MDLabel

class LiveViewScreen(MDScreen):
    def __init__(self, mainwid, **kwargs):
        super().__init__(**kwargs)
        self.mainwid = mainwid 
        # Create layout
        layout = MDBoxLayout(orientation='vertical')
        
        # Add toolbar
        toolbar = MDTopAppBar(
            title="Live Camera Feeds",
            right_action_items=[
                ["view-grid", lambda x: self.toggle_view("grid")],
                ["view-list", lambda x: self.toggle_view("list")]
            ]
        )
        
        # Add content
        content = MDLabel(
            text="Live Camera",
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
