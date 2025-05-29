import copy
import random


class NetworkGenerator:
    def __init__(self, num_points=0, x_range=(0, 20), y_range=(0, 20)): # TODO use real latitude and altitude
        """
        Initialize the random position generator.

        :param num_points: Number of points to generate.
        :param x_range: Tuple representing the range for x-coordinates (min_x, max_x).
        :param y_range: Tuple representing the range for y-coordinates (min_y, max_y).
        """
        self.num_points = num_points
        self.x_range = x_range
        self.y_range = y_range
        self.points = []

    def generate_servers_positions(self):
        """
        Generate random points within the specified ranges.

        :return: A list of tuples representing the random points [(x1, y1), (x2, y2), ...].
        """
        self.points = [
            (
                round(random.uniform(self.x_range[0], self.x_range[1]), 4),
                round(random.uniform(self.y_range[0], self.y_range[1]), 4)
            )
            for _ in range(self.num_points)
        ]
        return self.points

    def add_servers_positions(self, qty=10, server_locations=None, startpoint=0):
        for i in range(qty):
            self.points.append(
                (
                    server_locations[startpoint+i]
                )
            )

    # def append_servers_positions(self, i, old_servers_positions=None, qty=10, server_locations=None):
    #     has_changed = False
    #     if (old_servers_positions is not None) and (i == 0):
    #         servers_loc = old_servers_positions
    #     else:
    #         if old_servers_positions is None:
    #             servers_loc = [
    #                 (
    #                     server_locations[i]
    #                 )
    #                 for i in range(self.num_points)
    #             ]
    #         else:
    #             startpoint = len(old_servers_positions)
    #             for i in range(qty):
    #                 old_servers_positions.append(
    #                     (
    #                         server_locations[startpoint+i]
    #                     )
    #                 )
    #             servers_loc = old_servers_positions
    #         has_changed = True
    #
    #     return servers_loc, has_changed

    def append_servers_positions(self, i, old_servers_positions=None, qty=10, server_locations=None):
        has_changed = False
        if (old_servers_positions is not None) and (i == 0):
            servers_loc = old_servers_positions
        else:
            if old_servers_positions is None:
                print(f'server_locations={server_locations}')
                servers_loc = [
                    (
                        server_locations[j]
                    )
                    for j in range(qty)
                ]
            else:
                startpoint = len(old_servers_positions)
                new_serv_loc = copy.deepcopy(old_servers_positions)
                for j in range(qty):
                    new_serv_loc.append(
                        (
                            server_locations[startpoint+j]
                        )
                    )
                servers_loc = new_serv_loc
            has_changed = True

        return servers_loc, has_changed

# # Example Usage
# if __name__ == "__main__":
#     num_points = 3
#     x_range = (0, 50)  # Range for x-coordinates
#     y_range = (0, 50)  # Range for y-coordinates
#
#     generator = RandomPositionGenerator(num_points, x_range, y_range)
#     points = generator.generate_servers_positions()
#     print(points)
#
#     generator.add_servers_positions()
#     print(points)
