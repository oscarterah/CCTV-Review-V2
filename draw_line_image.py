def set_line_func(self):
    def on_touch_down(self, touch, *args):
        if self.my_card.collide_point(*touch.pos):
            self.line_coordinates = [touch.pos]

    def on_touch_move(self, touch, *args):
        if self.my_card.collide_point(*touch.pos):
            with self.my_card.canvas:
                Line(points=self.flatten_coordinates(self.line_coordinates + [touch.pos]), width=2)

            self.line_coordinates.append(touch.pos)

    def on_touch_up(self, touch, *args):
        if self.my_card.collide_point(*touch.pos):
            self.line_coordinates.append(touch.pos)

            #self.line_coordinates.append(self.line_coordinates)

            first_coordinate = self.line_coordinates[0]
            last_coordinate = self.line_coordinates[-1]

            x1, y1 = first_coordinate
            x2, y2 = last_coordinate
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            my_cords = [x1, y1, x2, y2]
            print(my_cords)
            self.my_coords = my_cords

    def flatten_coordinates(self, coordinates):
        return [coord for point in coordinates for coord in point]
    self.image_widget.bind(on_touch_down=self.on_touch_down, on_touch_move=self.on_touch_move, on_touch_up=self.on_touch_up)

