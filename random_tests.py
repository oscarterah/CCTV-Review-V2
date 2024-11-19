# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 02:32:18 2024

@author: oscar
"""

# #%%
# from kivy.lang import Builder
# from kivy.properties import StringProperty
# from kivy.uix.screenmanager import Screen

# from kivymd.icon_definitions import md_icons
# from kivymd.app import MDApp
# from kivymd.uix.list import OneLineIconListItem



# Builder.load_string(
#     '''
# #:import images_path kivymd.images_path


# <CustomOneLineIconListItem>

#     IconLeftWidget:
#         icon: root.icon


# <PreviousMDIcons>

#     MDBoxLayout:
#         orientation: 'vertical'
#         spacing: dp(10)
#         padding: dp(20)

#         MDBoxLayout:
#             adaptive_height: True

#             MDIconButton:
#                 icon: 'magnify'

#             MDTextField:
#                 id: search_field
#                 hint_text: 'Search icon'
#                 on_text: root.set_list_md_icons(self.text, True)

#         RecycleView:
#             id: rv
#             key_viewclass: 'viewclass'
#             key_size: 'height'

#             RecycleBoxLayout:
#                 padding: dp(10)
#                 default_size: None, dp(48)
#                 default_size_hint: 1, None
#                 size_hint_y: None
#                 height: self.minimum_height
#                 orientation: 'vertical'
# '''
# )


# class CustomOneLineIconListItem(OneLineIconListItem):
#     icon = StringProperty()


# class PreviousMDIcons(Screen):

#     def set_list_md_icons(self, text="", search=False):
#         '''Builds a list of icons for the screen MDIcons.'''

#         def add_icon_item(name_icon):
#             self.ids.rv.data.append(
#                 {
#                     "viewclass": "CustomOneLineIconListItem",
#                     "icon": name_icon,
#                     "text": name_icon,
#                     "callback": lambda x: x,
#                 }
#             )

#         self.ids.rv.data = []
#         for name_icon in md_icons.keys():
#             if search:
#                 if text in name_icon:
#                     add_icon_item(name_icon)
#             else:
#                 add_icon_item(name_icon)


# class MainApp(MDApp):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.screen = PreviousMDIcons()

#     def build(self):
#         return self.screen

#     def on_start(self):
#         self.screen.set_list_md_icons()


# MainApp().run()
#%%
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.navigationrail import MDNavigationRail, MDNavigationRailItem
from kivy.lang import Builder

KV = '''
MDBoxLayout:
    orientation: 'horizontal'
    
    MDNavigationRail:
        id: rail
        type: "selected"
        md_bg_color: app.theme_cls.bg_dark
        selected_color_background: app.theme_cls.primary_light
        ripple_color_item: app.theme_cls.primary_dark
        on_item_release: app.switch_screen(*args)
        
        MDNavigationRailItem:
            text: "Home"
            icon: "home"
            
        MDNavigationRailItem:
            text: "Profile"
            icon: "account"
            
        MDNavigationRailItem:
            text: "Settings"
            icon: "cog"
    
    MDScreenManager:
        id: screen_manager
        
        MDScreen:
            name: 'home'
            MDLabel:
                text: 'Home Screen'
                halign: 'center'
                
        MDScreen:
            name: 'profile'
            MDLabel:
                text: 'Profile Screen'
                halign: 'center'
                
        MDScreen:
            name: 'settings'
            MDLabel:
                text: 'Settings Screen'
                halign: 'center'
'''

class NavigationRailApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Purple"
        self.theme_cls.theme_style = "Dark"
        return Builder.load_string(KV)
    
    def switch_screen(self, instance_navigation_rail, instance_navigation_rail_item):
        # Get the screen name from the navigation item's text
        screen_name = instance_navigation_rail_item.text.lower()
        # Switch to the corresponding screen
        self.root.ids.screen_manager.current = screen_name

if __name__ == '__main__':
    NavigationRailApp().run()

#%%
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.navigationrail import MDNavigationRail, MDNavigationRailItem
from kivy.lang import Builder

KV = '''
MDBoxLayout:
    orientation: 'horizontal'
    
    MDNavigationRail:
        id: rail
        type: "selected"
        md_bg_color: app.theme_cls.bg_dark
        selected_color_background: app.theme_cls.primary_light
        ripple_color_item: app.theme_cls.primary_dark
        on_item_release: app.switch_screen(*args)
        
        MDNavigationRailItem:
            text: "Live View"
            icon: "cctv"
            
        MDNavigationRailItem:
            text: "Incidents"
            icon: "alert-circle"
            
        MDNavigationRailItem:
            text: "Analytics"
            icon: "chart-box"
            
        MDNavigationRailItem:
            text: "Cameras"
            icon: "video"
            
        MDNavigationRailItem:
            text: "Detection Config"
            icon: "cog-box"
            
        MDNavigationRailItem:
            text: "Alerts"
            icon: "bell"
            
        MDNavigationRailItem:
            text: "Archive"
            icon: "archive"
            
        MDNavigationRailItem:
            text: "Reports"
            icon: "file-chart"
            
        MDNavigationRailItem:
            text: "Users"
            icon: "account-group"
            
        MDNavigationRailItem:
            text: "Settings"
            icon: "cog"
    
    MDScreenManager:
        id: screen_manager
        
        MDScreen:
            name: 'live_view'
            MDBoxLayout:
                orientation: 'vertical'
                MDTopAppBar:
                    title: "Live Camera Feeds"
                    right_action_items: [["view-grid", "Grid View"], ["view-list", "List View"]]
                ScrollView:
                    MDGridLayout:
                        cols: 2
                        padding: dp(10)
                        spacing: dp(10)
                        # Add your camera feed widgets here
        
        MDScreen:
            name: 'incidents'
            MDBoxLayout:
                orientation: 'vertical'
                MDTopAppBar:
                    title: "Detected Incidents"
                    right_action_items: [["filter", "Filter"], ["sort", "Sort"]]
                MDList:
                    # Add incident items here
        
        MDScreen:
            name: 'analytics'
            MDBoxLayout:
                orientation: 'vertical'
                MDTopAppBar:
                    title: "Analytics Dashboard"
                MDTabs:
                    # Add analytics tabs and graphs here
        
        MDScreen:
            name: 'cameras'
            MDBoxLayout:
                orientation: 'vertical'
                MDTopAppBar:
                    title: "Camera Management"
                    right_action_items: [["plus", "Add Camera"]]
                MDList:
                    # Add camera list items here
        
        MDScreen:
            name: 'detection_config'
            MDBoxLayout:
                orientation: 'vertical'
                MDTopAppBar:
                    title: "Detection Configuration"
                MDList:
                    # Add detection settings here
        
        MDScreen:
            name: 'alerts'
            MDBoxLayout:
                orientation: 'vertical'
                MDTopAppBar:
                    title: "Alert Management"
                MDList:
                    # Add alert items here
        
        MDScreen:
            name: 'archive'
            MDBoxLayout:
                orientation: 'vertical'
                MDTopAppBar:
                    title: "Video Archive"
                    right_action_items: [["calendar", "Select Date"], ["filter", "Filter"]]
                MDList:
                    # Add archived footage items here
        
        MDScreen:
            name: 'reports'
            MDBoxLayout:
                orientation: 'vertical'
                MDTopAppBar:
                    title: "Reports"
                    right_action_items: [["file-export", "Export"], ["printer", "Print"]]
                MDList:
                    # Add report items here
        
        MDScreen:
            name: 'users'
            MDBoxLayout:
                orientation: 'vertical'
                MDTopAppBar:
                    title: "User Management"
                    right_action_items: [["account-plus", "Add User"]]
                MDList:
                    # Add user list items here
        
        MDScreen:
            name: 'settings'
            MDBoxLayout:
                orientation: 'vertical'
                MDTopAppBar:
                    title: "Settings"
                MDList:
                    # Add settings items here
'''

class CCTVAIApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "BlueGray"
        self.theme_cls.theme_style = "Dark"
        return Builder.load_string(KV)
    
    def switch_screen(self, instance_navigation_rail, instance_navigation_rail_item):
        # Convert screen name to lowercase and replace spaces with underscores
        screen_name = instance_navigation_rail_item.text.lower().replace(" ", "_")
        self.root.ids.screen_manager.current = screen_name

if __name__ == '__main__':
    CCTVAIApp().run()
#%%