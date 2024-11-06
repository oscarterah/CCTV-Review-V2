from kivymd.app import MDApp
from kivymd.uix.datatables import MDDataTable
from kivymd.uix.button import MDRaisedButton
from kivy.uix.boxlayout import BoxLayout

class ClickableTableApp(MDApp):
    def build(self):
        self.data_table = MDDataTable(
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            size_hint=(0.9, 0.6),
            column_data=[
                ("Column 1", dp(30)),
                ("Column 2", dp(30)),
                ("Column 3", dp(30)),
            ],
            row_data=[
                ("Row 1", "Data 1", "Data A"),
                ("Row 2", "Data 2", "Data B"),
                ("Row 3", "Data 3", "Data C"),
            ],
            check=True,
            use_pagination=True,
            rows_num=10,
            pagination_menu_height=300,
            elevation=2,
        )
        self.data_table.bind(on_row_press=self.on_row_press)

        button = MDRaisedButton(
            text="Show Table",
            pos_hint={'center_x': 0.5, 'center_y': 0.2},
            on_release=self.show_table
        )

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(button)

        return layout

    def on_row_press(self, instance_table, instance_row):
        # Handle the row press event
        print("Row press:", instance_row.text)

    def show_table(self, instance_button):
        # Show the table when the button is clicked
        self.data_table.open()

if __name__ == "__main__":
    ClickableTableApp().run()
