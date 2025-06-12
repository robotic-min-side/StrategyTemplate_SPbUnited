"""High-level strategy code"""

# !v DEBUG ONLY
from time import time
import math
import numpy as np

from bridge import const
from bridge.auxiliary import aux, fld, rbt
from bridge.const import State as GameStates
from bridge.router.base_actions import Action, Actions, KickActions
np.set_printoptions(threshold=np.inf)  # выводить весь массив полностью

class Strategy:
    """Main class of strategy"""

    def __init__(
            self,
    ) -> None:
        self.we_active = False
        self.centered_coordinates_result = self.process_coordinates_centered()
        self.goal_coords = self.generate_line_coords(-2150, -350, -2150, 350, 10)

    def process(self, field: fld.Field) -> list[Action]:
        """Game State Management"""
        if field.game_state not in [GameStates.KICKOFF, GameStates.PENALTY]:
            if field.active_team in [const.Color.ALL, field.ally_color]:
                self.we_active = True
            else:
                self.we_active = False

        actions: list[Action] = []
        for _ in range(const.TEAM_ROBOTS_MAX_COUNT):
            actions.append(Actions.Stop())

        if field.ally_color == const.COLOR:
            text = str(field.game_state) + "  we_active:" + str(self.we_active)
            field.strategy_image.print(aux.Point(600, 780), text, need_to_scale=False)
        match field.game_state:
            case GameStates.RUN:
                self.run(field, actions)
            case GameStates.TIMEOUT:
                pass
            case GameStates.HALT:
                return actions
            case GameStates.PREPARE_PENALTY:
                pass
            case GameStates.PENALTY:
                pass
            case GameStates.PREPARE_KICKOFF:
                pass
            case GameStates.KICKOFF:
                pass
            case GameStates.FREE_KICK:
                pass
            case GameStates.STOP:
                pass

        return actions

    def is_point_in_circle(self, point, center, radius):
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return dx * dx + dy * dy <= radius * radius

    def get_circles_containing_point(self, circles, point_mm):
        circles_containing_point = []
        for circle in circles:
            if self.is_point_in_circle(point_mm, circle['center'], circle['radius']):
                circles_containing_point.append(circle['name'])
        return sorted(circles_containing_point)

    def determine_polygon(self, circles_in):
        # Сопоставление списков кругов с номерами полигонов
        mapping = {
            (1,): 1,
            (1,2,3): 1,
            (2,3): 2,
            (2,3,6): 3,
            (2,3,4): 4,
            (2,4): 5,
            (2,): 6,
            (1,2): 7,
            (2,3,5): 8,
            (3,5): 9,
            (3,): 10,
            (1,3): 11,
            (2,3,4,6): 12,
            (2,3,5,6): 13,
        }
        return mapping.get(tuple(circles_in), 0)

    def get_polygon_for_point(self, point_mm):
        field_size_mm = (4500, 3000)
        half_field_size_mm = (field_size_mm[0] / 2, field_size_mm[1] / 2)

        circles = [
            {
                'name': 1,
                'center': (half_field_size_mm[0], 0),
                'radius': field_size_mm[0] / 2,
            },
            {
                'name': 2,
                'center': (-field_size_mm[1] * 1.8, field_size_mm[0] * 1.2),
                'radius': field_size_mm[0] * 1.7,
            },
            {
                'name': 3,
                'center': (-field_size_mm[1] * 1.8, -field_size_mm[0] * 1.2),
                'radius': field_size_mm[0] * 1.7,
            },
            {
                'name': 4,
                'center': (-half_field_size_mm[0], half_field_size_mm[0]),
                'radius': half_field_size_mm[0],
            },
            {
                'name': 5,
                'center': (-half_field_size_mm[0], -half_field_size_mm[0]),
                'radius': half_field_size_mm[0],
            },
            {
                'name': 6,
                'center': (-half_field_size_mm[0] + 100, 0),
                'radius': field_size_mm[0] / 3.5,
            },
        ]

        circles_in = self.get_circles_containing_point(circles, point_mm)
        polygon = self.determine_polygon(circles_in)
        return polygon

    def create_circles(self, field_w, field_h, scale, circles_data):
        """
        circles_data — список кортежей вида ((x, y), radius),
        где координаты и радиус уже заданы и будут масштабированы.
        """
        circles = [
            (
                (x / scale, y / scale),
                radius / scale
            )
            for (x, y), radius in circles_data
        ]
        return circles

    def scene_sdf(self, p, circles):
        return min(math.hypot(p[0] - c[0], p[1] - c[1]) - r for c, r in circles)

    def calculate_rays_min_diameters(self, origin_point, circles, scale, max_dist, surf_dist):
        collision_local = np.zeros(360, dtype=bool)
        min_diameters_local = np.full(360, float('inf'), dtype=np.float32)

        max_steps = 500
        step_limit = max_dist / scale

        for angle in range(360):
            theta = math.radians(angle)
            dir_v = (math.cos(theta), math.sin(theta))
            pos = np.array(origin_point, dtype=float)
            total_dist = 0.0
            min_d = float('inf')
            collided = False

            for _ in range(max_steps):
                d = self.scene_sdf((pos[0], pos[1]), circles)
                if d < surf_dist / scale:
                    min_d = 0.0
                    collided = True
                    break
                if total_dist > step_limit:
                    break
                min_d = min(min_d, d)
                step_size = max(d, 0.5)
                pos += np.array(dir_v) * step_size
                total_dist += step_size

            min_diameters_local[angle] = max(0.0, 2 * min_d * scale) if min_d != float('inf') else 0.0
            collision_local[angle] = collided

        return min_diameters_local, collision_local

    def fill_grid_from_angles_vectorized(self, origin_point, min_diameters_local, collision_local, w, h):
        grid_diameters_local = np.zeros((h, w), dtype=np.float32)

        ys = np.arange(0, h)
        xs = np.arange(0, w)
        X, Y = np.meshgrid(xs, ys)

        dx = X - origin_point[0]
        dy = Y - origin_point[1]

        angles = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        angle_indices = np.round(angles).astype(int) % 360

        collided_pixels = collision_local[angle_indices]
        grid_diameters_local[:] = np.where(collided_pixels, -1, min_diameters_local[angle_indices])

        return grid_diameters_local

    def run_raymarching_ckeck_pass(self, circles_data,
        field_w=4500, field_h=3000, scale=10,
        max_dist=5000.0, surf_dist=1.0, origin=None,
    ):
        w, h = field_w // scale, field_h // scale

        circles = self.create_circles(field_w, field_h, scale, circles_data)
        if origin is None:
            origin = (w // 2, h // 2)
        min_diameters_local, collision_local = self.calculate_rays_min_diameters(
            origin, circles, scale, max_dist, surf_dist
        )
        grid = self.fill_grid_from_angles_vectorized(origin, min_diameters_local, collision_local, w, h)
        grid = grid[::-1]
        return grid

    def process_coordinates_centered(self, x_range=4500, y_range=3000, step=10):
        x_half = x_range // 2
        y_half = y_range // 2

        x_coords = np.arange(-x_half, x_half, step)  # например, от -2250 до 2240
        y_coords = np.arange(-y_half, y_half, step)  # например, от -1500 до 1490

        results = np.empty([len(y_coords), len(x_coords)])

        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                results[i, j] = self.get_polygon_for_point([x, y])

        results = results[::-1]
        return results

    def create_gradient_array_with_decay(self, width, height, center_x, center_y, max_value=500, decay_coef=1.0):
        x = np.arange(width)
        y = np.arange(height)
        xv, yv = np.meshgrid(x, y, indexing='xy')

        dist = np.sqrt((xv - center_x)**2 + (yv - center_y)**2)

        max_dist = np.sqrt(width**2 + height**2)

        # Применяем коэффициент убавления к расстоянию (умножаем dist)
        adjusted_dist = dist * decay_coef

        # Градиент убывает от max_value в центре до 0, при dist = max_dist / decay_coef
        values = max_value * (1 - adjusted_dist / max_dist)
        values[values < 0] = 0

        return values

    def convert_coords(self, x_old, y_old, width, height):
        x_new = x_old - width / 2
        y_new = height / 2 - y_old
        return x_new, y_new

    def raymarch_min_distances(self, points, obstacles, direction_point, max_steps=100, surf_dist=1.0):
        """
        points: список или массив точек [(x1, y1), (x2, y2), ...] — origin для каждого луча
        obstacles: список из трёх кортежей ((cx, cy), radius)
        direction_point: точка (x, y), задающая направление лучей для всех origin
        max_steps: макс. число шагов raymarching
        surf_dist: порог расстояния до поверхности
        
        Возвращает список минимальных безопасных диаметров (2*d) для каждой точки.
        """
        def scene_sdf(p):
            return min(math.hypot(p[0] - c[0], p[1] - c[1]) - r for c, r in obstacles)

        # Вычисляем единичный вектор направления от каждой точки к direction_point
        def normalize(v):
            length = math.hypot(v[0], v[1])
            if length < 1e-8:
                return (0.0, 0.0)
            return (v[0] / length, v[1] / length)

        dir_v = normalize((direction_point[0] - points[0][0], direction_point[1] - points[0][1]))
        # Если нужно, можно менять направление для каждой точки отдельно, но по условию — общее

        results = []
        for origin in points:
            # Направление для каждого origin — от origin к direction_point
            dir_v = normalize((direction_point[0] - origin[0], direction_point[1] - origin[1]))
            pos = np.array(origin, dtype=float)
            min_d = float('inf')

            for _ in range(max_steps):
                d = scene_sdf((pos[0], pos[1]))
                if d < surf_dist:
                    min_d = 0.0
                    break
                if d > 0:
                    min_d = min(min_d, 2 * d)
                else:
                    min_d = 0.0
                    break
                pos += np.array(dir_v) * d

            results.append(min_d if min_d != float('inf') else 0.0)
        return results

    def generate_line_coords(self, x1, y1, x2, y2, step):
        # Вычисляем длину линии
        length = np.hypot(x2 - x1, y2 - y1)
        # Количество точек с учётом шага
        num_points = int(length / step) + 1
        # Линейно интерполируем координаты
        x_coords = np.linspace(x1, x2, num_points)
        y_coords = np.linspace(y1, y2, num_points)
        # Создаём список кортежей координат
        coords = list(zip(x_coords, y_coords))
        return coords

    def check_goal_opportunity(self, first_attack_rbt, enemies):
        """Проверка на гол"""
        circles_data = [
            ((enemies[0].x, enemies[0].y), 90),
            ((enemies[1].x, enemies[1].y), 90),
            ((enemies[2].x, enemies[2].y), 90),
        ]
        distances = self.raymarch_min_distances(self.goal_coords, circles_data, [first_attack_rbt.x, first_attack_rbt.y])
        if np.max(distances) > 150:
            goal_id = np.argmax(distances)
            goal_kick_coord = goal_id*10-350
            return goal_kick_coord
        return None

    def find_best_pass_point(self, first_attack_rbt, second_attack_rbt, enemies):
        """Проверка на место принятия паса"""
        circles_data = [
            ((enemies[0].x + 2250, enemies[0].y + 1500), 90),
            ((enemies[1].x + 2250, enemies[1].y + 1500), 90),
        ]
        possible_pass_points = self.run_raymarching_ckeck_pass(
            circles_data, 
            origin=[(first_attack_rbt.x + 2250)/10, (first_attack_rbt.y + 1500)/10]
        )
        possible_goal_points = self.run_raymarching_ckeck_pass(
            circles_data, 
            origin=[0, (1500)/10]
        )
        gradient_array = self.create_gradient_array_with_decay(
            450, 300, 
            (second_attack_rbt.x + 2250)/10, 
            (second_attack_rbt.y + 1500)/10, 
            max_value=500, 
            decay_coef=2
        )

        gradient_array = gradient_array[::-1]

        summ = self.replace_minus_one(possible_pass_points) + self.replace_minus_one(possible_goal_points) + self.replace_minus_one(gradient_array)

        summ = self.replace_first_10_elements(summ)
        
        np.savetxt('array.csv', summ , delimiter=',', fmt='%d')

        max_idx_flat = np.argmax(summ)
        max_idx_2d = np.unravel_index(max_idx_flat, summ.shape)
        x_ind, y_inx = self.convert_coords(max_idx_2d[1], max_idx_2d[0], 450, 300)
        return [x_ind*10, y_inx*10]
    
    def replace_minus_one(self, arr):
        arr = np.array(arr)  # на случай, если вход не является numpy-массивом
        arr[arr == -1] = -10000
        return arr

    def replace_first_10_elements(self, arr):
        arr = np.array(arr)  # на случай, если вход не является numpy-массивом
        arr[:, :10] = -10000
        return arr
    

    def calculate_velocities(self, cx, cy, t, base_radius=5, radius_amplitude=2, angular_speed=1):
        """
        cx, cy - координаты центра вращения
        t - текущее время (или шаг)
        base_radius - средний радиус круга
        radius_amplitude - амплитуда изменения радиуса (подъезд/отъезд)
        angular_speed - угловая скорость вращения (рад/с)

        Возвращает список из трех кортежей (vx, vy) - скорости для каждого робота.
        """
        velocities = []
        num_robots = 3
        for i in range(num_robots):
            angle = angular_speed * t + 2 * math.pi * i / num_robots
            radius = base_radius + radius_amplitude * math.sin(angular_speed * t)
            
            # Производная радиуса по времени
            dr_dt = radius_amplitude * angular_speed * math.cos(angular_speed * t)
            
            # Скорость по осям:
            # vx = dr/dt * cos(angle) - radius * angular_speed * sin(angle)
            # vy = dr/dt * sin(angle) + radius * angular_speed * cos(angle)
            vx = dr_dt * math.cos(angle) - radius * angular_speed * math.sin(angle)
            vy = dr_dt * math.sin(angle) + radius * angular_speed * math.cos(angle)
            
            velocities.append((vx, vy))
        
        return velocities













    def run(self, field: fld.Field, actions: list[Action]) -> None:
        """
        Assigning roles to robots and managing them
            roles - robot roles sorted by priority
            robot_roles - list of robot id and role matches
        """

        ball = field.ball.get_pos()
        rbt_allies_0 = field.allies[0].get_pos()
        rbt_allies_1 = field.allies[1].get_pos()
        rbt_allies_2 = field.allies[2].get_pos()
        rbt_enemies_0 = field.enemies[0].get_pos()
        rbt_enemies_1 = field.enemies[1].get_pos()
        rbt_enemies_2 = field.enemies[2].get_pos()
        print()
        if aux.dist(ball, rbt_allies_1) >= aux.dist(ball, rbt_allies_2):
            first_attack_rbt = rbt_allies_2
            second_attack_rbt = rbt_allies_1
            first_attack_rbt_id = 2
            second_attack_rbt_id = 1
        else:
            first_attack_rbt = rbt_allies_1
            second_attack_rbt = rbt_allies_2
            first_attack_rbt_id = 1
            second_attack_rbt_id = 2

        #print(self.centered_coordinates_result)

        
        #пасы друг другу
        #actions[first_attack_rbt_id] = Actions.Kick(second_attack_rbt, is_pass=True)
        #if field.is_ball_moves_to_point(second_attack_rbt) == True:
        #    actions[second_attack_rbt_id] = Actions.GoToPoint(aux.closest_point_on_line(ball, ball + field.ball.get_vel(), second_attack_rbt, "R"), aux.angle_to_point(second_attack_rbt, ball))
        

        #field.strategy_image.draw_dot(aux.Point(1000,1000), [255,255,0], 10) отрисовка точки

        #circles_data = [
        #    ((rbt_enemies_1.x + 2250, rbt_enemies_1.y + 1500), 90),
        #    ((rbt_enemies_2.x + 2250, rbt_enemies_2.y + 1500), 90),
        #]   
        #np.savetxt('array.csv', self.run_raymarching_ckeck_pass(circles_data, origin=[(ball.x + 2250)/10, (ball.y + 1500)/10]), delimiter=',', fmt='%d') это для вызова рэймарчинга


        #ball_polygon = self.get_polygon_for_point([ball.x, ball.y]) узнать полигон мяча
        
        #print(self.check_goal_opportunity(first_attack_rbt, [rbt_enemies_0, rbt_enemies_1, rbt_enemies_2]))
        
        #print(self.find_best_pass_point(first_attack_rbt, second_attack_rbt, [rbt_enemies_1, rbt_enemies_2]))






        find_pass = self.find_best_pass_point(first_attack_rbt, second_attack_rbt, [rbt_enemies_1, rbt_enemies_2])
        
        if aux.dist(first_attack_rbt, ball) < 200:
            check_goal = self.check_goal_opportunity(first_attack_rbt, [rbt_enemies_0, rbt_enemies_1, rbt_enemies_2])
            if check_goal != None:
                actions[first_attack_rbt_id] = Actions.Kick(aux.Point(-2250, check_goal), is_pass=True)
                actions[second_attack_rbt_id] = Actions.GoToPoint(aux.Point(find_pass[0], find_pass[1]), aux.angle_to_point(second_attack_rbt, ball))
            else:
                if abs(second_attack_rbt.x - find_pass[0]) < 200 and abs(second_attack_rbt.y - find_pass[1]) < 200:
                    actions[first_attack_rbt_id] = Actions.Kick(second_attack_rbt, is_pass=True)
                    
                elif field.is_ball_moves_to_point(second_attack_rbt) == True:
                    actions[second_attack_rbt_id] = Actions.GoToPoint(aux.closest_point_on_line(ball, ball + field.ball.get_vel(), second_attack_rbt, "R"), aux.angle_to_point(second_attack_rbt, ball))
                else:
                    actions[second_attack_rbt_id] = Actions.GoToPoint(aux.Point(find_pass[0], find_pass[1]), aux.angle_to_point(second_attack_rbt, ball))
        else:
            actions[first_attack_rbt_id] = Actions.GoToPoint(ball, aux.angle_to_point(first_attack_rbt, ball))
            actions[second_attack_rbt_id] = Actions.GoToPoint(aux.Point(find_pass[0], find_pass[1]), aux.angle_to_point(second_attack_rbt, ball))
        


        '''
        circles_data = [
            ((rbt_enemies_1.x + 2250, rbt_enemies_1.y + 1500), 90),
            ((rbt_enemies_2.x + 2250, rbt_enemies_2.y + 1500), 90),
        ]
        np.savetxt('array.csv', self.run_raymarching_ckeck_pass(
            circles_data, 
            origin=[0, (1500)/10]
        ), delimiter=',', fmt='%d')
        '''


        #   

        field.strategy_image.draw_dot(aux.Point(find_pass[0], find_pass[1]), [255,0,255], 25)




















        if field.is_ball_moves_to_point(rbt_allies_0) == True:
            actions[0] = Actions.GoToPoint(aux.get_line_intersection(ball, ball + field.ball.get_vel(), aux.Point(2300, 400), aux.Point(2300, -400), "LL"), aux.angle_to_point(rbt_allies_0, ball))
        else:
            actions[0] = Actions.GoToPoint(aux.Point(2200), aux.angle_to_point(rbt_allies_0, ball))