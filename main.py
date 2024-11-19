import os, sys
import kivy
from kivy.resources import resource_add_path, resource_find
from MainApp import MainApp

if __name__ == '__main__':
    if hasattr(sys, '_MEIPASS'):
        resource_add_path(os.path.join(sys._MEIPASS))

    MainApp().run()
