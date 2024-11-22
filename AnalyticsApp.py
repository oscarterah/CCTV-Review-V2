# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:45:59 2024

@author: oscar
"""
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.screen import MDScreen
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.label import MDLabel




class AnalyticsScreen(MDScreen):
    def __init__(self, mainwid, **kwargs):
        super().__init__()
        self.mainwid = mainwid
        
        
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
            text="Analytics",
            halign='center'
        )
        
        layout.add_widget(toolbar)
        layout.add_widget(content)
        self.add_widget(layout)

