from abc import ABC, abstractmethod
from typing import NamedTuple, Union, Optional, Tuple

import jax
import jax.numpy as jnp
import os
import jax.tree_util as jtu
import numpy as np
from jax import random, vmap
from jaxtyping import PyTree


class PendulumParams(NamedTuple):
    m: jax.Array = jnp.array(1.0)
    l: jax.Array = jnp.array(1.0)
    g: jax.Array = jnp.array(9.81)
    nu: jax.Array = jnp.array(0.0)
    c_d: jax.Array = jnp.array(0.0)


class CarParams(NamedTuple):
    """
    d_f, d_r : Represent grip of the car. Range: [0.015, 0.025]
    b_f, b_r: Slope of the pacejka. Range: [2.0 - 4.0].

    delta_limit: [0.3 - 0.5] -> Limit of the steering angle.

    c_m_1: Motor parameter. Range [0.2, 0.5]
    c_m_1: Motor friction, Range [0.00, 0.007]
    c_f, c_r: [1.0 2.0] # motor parameters: source https://web.mit.edu/drela/Public/web/qprop/motor1_theory.pdf,
    https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/Embedded
    -Control-Systems/LectureNotes/6_Motor_Control.pdf # or look at:
    https://video.ethz.ch/lectures/d-mavt/2021/spring/151-0593-00L/00718f4f-116b-4645-91da-b9482164a3c7.html :
    lecture 2 part 2
    c_m_1: max current of motor: [0.2 - 0.5] c_m_2: motor resistance due to shaft: [0.01 - 0.15]

    c_rr: zero torque current: [0 0.1]
    """
    m: Union[jax.Array, float] = jnp.array(0.05)  # [0.04, 0.08]
    i_com: Union[jax.Array, float] = jnp.array(27.8e-6)  # [1e-6, 5e-6]
    l_f: Union[jax.Array, float] = jnp.array(0.03)  # [0.025, 0.05]
    l_r: Union[jax.Array, float] = jnp.array(0.035)  # [0.025, 0.05]
    g: Union[jax.Array, float] = jnp.array(9.81)
    d_f: Union[jax.Array, float] = jnp.array(0.02)  # [0.015, 0.025]
    c_f: Union[jax.Array, float] = jnp.array(1.2)  # [1.0, 2.0]
    b_f: Union[jax.Array, float] = jnp.array(2.58)  # [2.0, 4.0]
    d_r: Union[jax.Array, float] = jnp.array(0.017)  # [0.015, 0.025]
    c_r: Union[jax.Array, float] = jnp.array(1.27)  # [1.0, 2.0]
    b_r: Union[jax.Array, float] = jnp.array(3.39)  # [2.0, 4.0]
    c_m_1: Union[jax.Array, float] = jnp.array(0.2)  # [0.2, 0.5]
    c_m_2: Union[jax.Array, float] = jnp.array(0.05)  # [0.00, 0.007]
    c_rr: Union[jax.Array, float] = jnp.array(0.003)  # [0.001, 0.01]
    c_d: Union[jax.Array, float] = jnp.array(0.052)  # [0.01, 0.1]
    steering_limit: Union[jax.Array, float] = jnp.array(0.35)
    use_blend: Union[jax.Array, float] = jnp.array(0.0)  # 0.0 -> no blend (only kinematics), 1.0 -> (kinematics + dynamics)


class DynamicsModel(ABC):
    def __init__(self,
                 dt: float,
                 x_dim: int,
                 u_dim: int,
                 params: PyTree,
                 angle_idx: Optional[Union[int, jax.Array]] = None,
                 dt_integration: float = 0.01,
                 ):
        self.dt = dt
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.params = params
        self.angle_idx = angle_idx

        self.dt_integration = dt_integration
        assert dt >= dt_integration
        assert (dt / dt_integration - int(dt / dt_integration)) < 1e-4, 'dt must be multiple of dt_integration'
        self._num_steps_integrate = int(dt / dt_integration)

    def next_step(self, x: jax.Array, u: jax.Array, params: PyTree) -> jax.Array:
        def body(carry, _):
            q = carry + self.dt_integration * self.ode(carry, u, params)
            return q, None

        next_state, _ = jax.lax.scan(body, x, xs=None, length=self._num_steps_integrate)
        if self.angle_idx is not None:
            theta = next_state[self.angle_idx]
            sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
            next_state = next_state.at[self.angle_idx].set(jnp.arctan2(sin_theta, cos_theta))
        return next_state

    def ode(self, x: jax.Array, u: jax.Array, params) -> jax.Array:
        assert x.shape[-1] == self.x_dim and u.shape[-1] == self.u_dim
        return self._ode(x, u, params)

    @abstractmethod
    def _ode(self, x: jax.Array, u: jax.Array, params) -> jax.Array:
        pass

    def _split_key_like_tree(self, key: jax.random.PRNGKey):
        treedef = jtu.tree_structure(self.params)
        keys = jax.random.split(key, treedef.num_leaves)
        return jtu.tree_unflatten(treedef, keys)

    def sample_params_uniform(self, key: jax.random.PRNGKey, sample_shape: Union[int, Tuple[int]],
                              lower_bound: NamedTuple, upper_bound: NamedTuple):
        keys = self._split_key_like_tree(key)
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        return jtu.tree_map(lambda key, l, u: jax.random.uniform(key, shape=sample_shape + l.shape, minval=l, maxval=u),
                            keys, lower_bound, upper_bound)


class Pendulum(DynamicsModel):
    _metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, dt: float, params: PendulumParams = PendulumParams(), dt_integration: float = 0.005,
                 encode_angle: bool = True):
        super().__init__(dt=dt, x_dim=2, u_dim=1, params=params, angle_idx=0, dt_integration=dt_integration)
        self.encode_angle = encode_angle

        # attributes for rendering
        self.render_mode = 'human'
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        self.state = None
        self.last_u = None

    def _ode(self, x, u, params: PendulumParams):
        # x represents [theta in rad/s, theta_dot in rad/s^2]
        # u represents [torque]

        x0_dot = x[..., 1]
        # We add drag force: https://www.scirp.org/journal/paperinformation.aspx?paperid=73856
        f_drag_linear = - params.nu * x[..., 1] / (params.m * params.l)
        f_drag_second_order = - params.c_d / params.m * (x[..., 1]) ** 2
        f_drag = f_drag_linear + f_drag_second_order
        x1_dot = params.g / params.l * jnp.sin(x[..., 0]) + u[..., 0] / (params.m * params.l ** 2) + f_drag
        return jnp.stack([x0_dot, x1_dot], axis=-1)

    def next_step(self, x: jax.Array, u: jax.Array, params: PyTree, encode_angle: Optional[bool] = None) -> jax.Array:
        if encode_angle is None:
            encode_angle = self.encode_angle
        if encode_angle:
            assert x.shape[-1] == 3
            theta = jnp.arctan2(x[..., 0], x[..., 1])
            x_radian = jnp.stack([theta, x[..., -1]], axis=-1)
            theta_new, theta_dot_new = jnp.split(super().next_step(x_radian, u, params), 2, axis=-1)
            next_state = jnp.concatenate([jnp.sin(theta_new), jnp.cos(theta_new), theta_dot_new], axis=-1)
            assert next_state.shape == x.shape
            return next_state
        else:
            return super().next_step(x, u, params)

    def reset(self, key: jax.random.PRNGKey = jax.random.PRNGKey(34565)):
        theta_diff = jax.random.uniform(key, shape=(), minval=-0.1, maxval=0.1)
        theta = np.pi + theta_diff
        theta = jnp.arctan2(jnp.sin(theta), jnp.cos(theta))
        self.state = jnp.array([theta, 0.0])
        if self.encode_angle:
            return jnp.array([jnp.sin(theta), jnp.cos(theta), 0.0])
        else:
            return self.state

    def step(self, u: Union[jnp.array, float]):
        self.last_u = u
        self.state = self.next_step(self.state, u, self.params, encode_angle=False)
        if self.encode_angle:
            theta, theta_dot = self.state[..., 0], self.state[..., 1]
            return jnp.stack([jnp.sin(theta), jnp.cos(theta), theta_dot], axis=-1)
        else:
            return self.state

    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise RuntimeError("pygame is not installed, run `pip install pygame`")

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + jnp.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + jnp.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        from gym.envs.classic_control import pendulum
        fname = os.path.join(os.path.dirname(pendulum.__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * abs(float(self.last_u)) / 2, scale * float(abs(self.last_u)) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self._metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return jnp.transpose(
                jnp.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class RaceCar(DynamicsModel):

    """
    x = [x, y, theta, vel_x, vel_y, vel_theta]
    u = [steering_angle, throttle]
    """

    def __init__(self, dt, encode_angle: bool = False):
        super().__init__(dt=dt, x_dim=6, u_dim=2, params=CarParams(), angle_idx=2)
        self.encode_angle = encode_angle
        self.angle_idx = 2

    def next_step(self, x, u, params):
        if self.encode_angle:
            theta = jnp.arctan2(x[..., self.angle_idx], x[..., self.angle_idx + 1])

            x_reduced = jnp.concatenate([x[..., 0:self.angle_idx], jnp.atleast_1d(theta), x[..., self.angle_idx + 2:]], axis=-1)
            x_reduced = super().next_step(x_reduced, u, params)
            next_theta = jnp.atleast_1d(x_reduced[..., self.angle_idx])
            next_x = jnp.concatenate([x_reduced[..., 0:self.angle_idx], jnp.sin(next_theta), jnp.cos(next_theta),
                                x_reduced[..., self.angle_idx + 1:]], axis=-1)
        else:
            next_x = super().next_step(x, u, params)
        return next_x

    @staticmethod
    def _ode_kin(x, u, params: CarParams):
        p_x, p_y, theta, v_x = x[0], x[1], x[2], x[3]  # progress
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2
        c_d = params.c_d
        c_rr = params.c_rr
        delta, d = u[0], u[1]
        f_r_x = ((c_m_1 - c_m_2 * v_x) * d - c_rr - c_d * (v_x ** 2)) / m
        beta = jnp.arctan(l_r * jnp.tan(delta) / (l_r + l_f))
        p_x_dot = v_x * jnp.cos(beta)  # s_dot
        p_y_dot = v_x * jnp.sin(beta)  # d_dot
        w = v_x * jnp.sin(beta) / l_r
        p_g_x_dot = p_x_dot * jnp.cos(theta) - p_y_dot * jnp.sin(theta)
        p_g_y_dot = p_x_dot * jnp.sin(theta) + p_y_dot * jnp.cos(theta)
        dx_kin = jnp.asarray([p_g_x_dot, p_g_y_dot, w, f_r_x])
        return dx_kin

    def _accelerations(self, x, u, params: CarParams):
        i_com = params.i_com
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        d_f = params.d_f * params.g
        d_r = params.d_r * params.g
        c_f = params.c_f
        c_r = params.c_r
        b_f = params.b_f
        b_r = params.b_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2
        c_rr = params.c_rr

        c_d = params.c_d

        delta, d = u[0], u[1]

        alpha_f = -jnp.arctan(
            (w * l_f + v_y) /
            (v_x + 1e-6)
        ) + delta
        alpha_r = jnp.arctan(
            (w * l_r - v_y) /
            (v_x + 1e-6)
        )
        f_f_y = d_f * jnp.sin(c_f * jnp.arctan(b_f * alpha_f))
        f_r_y = d_r * jnp.sin(c_r * jnp.arctan(b_r * alpha_r))
        f_r_x = (c_m_1 - c_m_2 * v_x) * d - c_rr - c_d * (v_x ** 2)

        v_x_dot = (f_r_x - f_f_y * jnp.sin(delta) + m * v_y * w) / m
        v_y_dot = (f_r_y + f_f_y * jnp.cos(delta) - m * v_x * w) / m
        w_dot = (f_f_y * l_f * jnp.cos(delta) - f_r_y * l_r) / i_com

        acceleration = jnp.array([v_x_dot, v_y_dot, w_dot])
        return acceleration

    def _ode_dyn(self, x, u, params: CarParams):
        # state = [p_x, p_y, theta, v_x, v_y, w]. Velocities are in local coordinate frame.
        # Inputs: [\delta, d] -> \delta steering angle and d duty cycle of the electric motor.
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        p_x_dot = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        p_y_dot = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        theta_dot = w
        p_x_dot = jnp.array([p_x_dot, p_y_dot, theta_dot])

        accelerations = self._accelerations(x, u, params)

        x_dot = jnp.concatenate([p_x_dot, accelerations], axis=-1)
        return x_dot

    def _compute_dx_kin(self, x, u, params: CarParams):
        l_r = params.l_r
        l_f = params.l_f
        v_x = x[3]
        v_y = x[4]
        x_kin = jnp.asarray([x[0], x[1], x[2], jnp.sqrt(v_x ** 2 + v_y ** 2)])
        dxkin = self._ode_kin(x_kin, u, params)
        delta = u[0]
        beta = jnp.arctan(l_r * jnp.tan(delta) / l_f + l_r)
        d_v_x_state = dxkin[3] * jnp.cos(beta)  # V*cos(beta)
        d_v_y_state = dxkin[3] * jnp.sin(beta)  # V*sin(beta)
        w = dxkin[3] * jnp.arctan(delta) / (l_f + l_r)
        dx_kin_full = jnp.asarray([dxkin[0], dxkin[1], dxkin[2], d_v_x_state, d_v_y_state, w])
        return dx_kin_full

    def _compute_dx(self, x, u, params: CarParams):
        use_kin = params.use_blend <= 0.5
        v_x = x[3]
        blend_ratio = (v_x - 0.3) / 0.2
        lambda_blend = jnp.min(jnp.asarray([
            jnp.max(jnp.asarray([blend_ratio, 0])), 1])
        )
        dx_kin_full = self._compute_dx_kin(x, u, params)
        dx_dyn = self._ode_dyn(x=x, u=u, params=params)
        dx_blend = lambda_blend * dx_dyn + (1 - lambda_blend) * dx_kin_full
        dx = (1 - use_kin) * dx_blend + use_kin * dx_kin_full
        return dx

    def _ode(self, x, u, params: CarParams):
        """
        Using kinematic model with blending: https://arxiv.org/pdf/1905.05150.pdf
        Code based on: https://github.com/alexliniger/gym-racecar/
        """
        delta, d = u[0], u[1]
        delta = jnp.clip(delta, a_min=-params.steering_limit,
                         a_max=params.steering_limit)
        d = jnp.clip(d, a_min=0., a_max=1)  # throttle
        u = u.at[0].set(delta)
        u = u.at[1].set(d)
        v_x = x[3]
        idx = jnp.logical_and(v_x <= 0.1,
                              d <= (params.c_rr + params.c_d * v_x ** 2) / (params.c_m_1 - params.c_m_2 * v_x))

        def stop_acceleration_update(x, u, param: CarParams):
            return jnp.zeros_like(x)

        dx = jax.lax.cond(
            idx,
            stop_acceleration_update,
            self._compute_dx,
            x,
            u,
            params,
        )
        return dx


if __name__ == "__main__":
    pendulum = Pendulum(0.1)
    pendulum.next_step(x=jnp.array([0., 0., 0.]), u=jnp.array([1.0]), params=pendulum.params)

    upper_bound = PendulumParams(m=jnp.array(1.0), l=jnp.array(1.0), g=jnp.array(10.0), nu=jnp.array(1.0),
                                 c_d=jnp.array(1.0))
    lower_bound = PendulumParams(m=jnp.array(0.1), l=jnp.array(0.1), g=jnp.array(9.0), nu=jnp.array(0.1),
                                 c_d=jnp.array(0.1))
    key = jax.random.PRNGKey(0)
    keys = random.split(key, 4)
    params = vmap(pendulum.sample_params_uniform, in_axes=(0, None, None, None))(keys, 1, upper_bound, lower_bound)

    def simulate_car(init_pos=jnp.zeros(2), horizon=150):
        dt = 0.01
        car = RaceCar(dt)
        params = CarParams(use_blend=0.0)
        x = jnp.zeros(6)
        x_traj = jnp.zeros([horizon, 2])
        x = x.at[0:2].set(init_pos)
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for h in range(horizon):
            s = 0.35
            d = 1
            u = jnp.asarray([s, d])
            x = car.next_step(x, u, params)
            x_traj = x_traj.at[h, ...].set(x[0:2])

        ax.plot(x_traj[:, 0], x_traj[:, 1], label='Car Trajectory w/o blend')

        params = CarParams(use_blend=1.0)
        x = jnp.zeros(6)
        x_traj = jnp.zeros([horizon, 2])
        x = x.at[0:2].set(init_pos)
        for h in range(horizon):
            s = 0.35
            d = 1
            u = jnp.asarray([s, d])
            x = car.next_step(x, u, params)
            x_traj = x_traj.at[h, ...].set(x[0:2])

        ax.plot(x_traj[:, 0], x_traj[:, 1], label='Car Trajectory with blend')
        # plt.scatter(env._goal[0], env._goal[1], color='red', label='goal')
        plt.legend()
        plt.xlabel('x-distance in [m]')
        plt.ylabel('y-distance in [m]')
        plt.title("Simulation of Car for " + str(int(horizon)))
        plt.show()


    simulate_car()
