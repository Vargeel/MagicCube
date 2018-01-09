import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2


class gameEnv():
    def __init__(self, N = 3):

        self.facedict = {"U": 0, "D": 1, "F": 2, "B": 3, "R": 4, "L": 5}
        self.dictface = dict([(v, k) for k, v in self.facedict.items()])
        self.normals = [np.array([0., 1., 0.]), np.array([0., -1., 0.]),
                   np.array([0., 0., 1.]), np.array([0., 0., -1.]),
                   np.array([1., 0., 0.]), np.array([-1., 0., 0.])]
        # this xdirs has to be synchronized with the self.move() function

        self.xdirs = [np.array([1., 0., 0.]), np.array([1., 0., 0.]),
                 np.array([1., 0., 0.]), np.array([-1., 0., 0.]),
                 np.array([0., 0., -1.]), np.array([0, 0., 1.])]
        self.colordict = {"w": 0, "y": 1, "b": 2, "g": 3, "o": 4, "r": 5}
        self.pltpos = [(0., 1.05), (0., -1.05), (0., 0.), (2.10, 0.), (1.05, 0.), (-1.05, 0.)]
        self.labelcolor = "#7f00ff"
        self.N = N
        self.stickers = np.array([np.tile(i, (self.N, self.N)) for i in range(6)])
        self.stickercolors = ["w", "#ffcf00", "#00008f", "#009f0f", "#ff6f00", "#cf0000"]
        self.plasticcolor = "#1f1f1f"
        self.fontsize = 12. * (self.N / 5.)
        self.actions = 6 * self.N
        self.actions_list = range(self.actions)
        self.action_to_rot = []
        for f in range(6):
            for l in range(self.N):
                for d in [1]:
                    self.action_to_rot.append([f,l,d])

        return None

    def move(self, f, l, d):
        """
        Make a layer move of layer `l` parallel to face `f` through
        `d` 90-degree turns in the clockwise direction.  Layer `0` is
        the face itself, and higher `l` values are for layers deeper
        into the cube.  Use `d=3` or `d=-1` for counter-clockwise
        moves, and `d=2` for a 180-degree move..
        """
        i = self.facedict[f]
        l2 = self.N - 1 - l
        assert l < self.N
        ds = range((d + 4) % 4)
        if f == "U":
            f2 = "D"
            i2 = self.facedict[f2]
            for d in ds:
                self._rotate([(self.facedict["F"], range(self.N), l2),
                              (self.facedict["R"], range(self.N), l2),
                              (self.facedict["B"], range(self.N), l2),
                              (self.facedict["L"], range(self.N), l2)])
        if f == "D":
            return self.move("U", l2, -d)
        if f == "F":
            f2 = "B"
            i2 = self.facedict[f2]
            for d in ds:
                self._rotate([(self.facedict["U"], range(self.N), l),
                              (self.facedict["L"], l2, range(self.N)),
                              (self.facedict["D"], range(self.N)[::-1], l2),
                              (self.facedict["R"], l, range(self.N)[::-1])])
        if f == "B":
            return self.move("F", l2, -d)
        if f == "R":
            f2 = "L"
            i2 = self.facedict[f2]
            for d in ds:
                self._rotate([(self.facedict["U"], l2, range(self.N)),
                              (self.facedict["F"], l2, range(self.N)),
                              (self.facedict["D"], l2, range(self.N)),
                              (self.facedict["B"], l, range(self.N)[::-1])])
        if f == "L":
            return self.move("R", l2, -d)
        for d in ds:
            if l == 0:
                self.stickers[i] = np.rot90(self.stickers[i], 3)
            if l == self.N - 1:
                self.stickers[i2] = np.rot90(self.stickers[i2], 1)
        # print "moved", f, l, len(ds)
        return None

    def _rotate(self, args):
        """
        Internal function for the `move()` function.
        """
        a0 = args[0]
        foo = self.stickers[a0]
        a = a0
        for b in args[1:]:
            self.stickers[a] = self.stickers[b]
            a = b
        self.stickers[a] = foo
        return None

    def randomize(self, number):
        """
        Make `number` randomly chosen moves to scramble the cube.
        """
        for t in range(number):
            f = self.dictface[np.random.randint(6)]
            l = np.random.randint(self.N)
            d = 1
            self.move(f, l, d)
        _, d = self.checkGoal()
        if d :
            r = random.randint(0, self.actions -1)
            f, l, d = self.convert_action_into_rotation(r)
            self.move(f,l,d)

        return None

    def render_flat(self, ax):
        """
        Make an unwrapped, flat view of the cube for the `render()`
        function.  This is a map, not a view really.  It does not
        properly render the plastic and stickers.
        """
        for f, i in self.facedict.items():
            x0, y0 = self.pltpos[i]
            cs = 1. / self.N
            for j in range(self.N):
                for k in range(self.N):
                    ax.add_artist(Rectangle((x0 + j * cs, y0 + k * cs), cs, cs, ec=self.plasticcolor,
                                            fc=self.stickercolors[self.stickers[i, j, k]]))

        return None

    def render(self):
        """
        Visualize the cube in a standard layout, including a flat,
        unwrapped view and three perspective views.
        """
        xlim = (-1.2, 3.2)
        ylim = (-1.2, 2.2)
        fig = plt.figure(figsize=((xlim[1] - xlim[0]) * self.N / 5., (ylim[1] - ylim[0]) * self.N / 5.))
        ax = fig.add_axes((0, 0, 1, 1), frameon=False,
                          xticks=[], yticks=[])
        self.render_flat(ax)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        fig.canvas.draw()

        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return cv2.resize(data,(data.shape[1]/2,data.shape[0]/2))

    def reset(self, max_steps):
        self.__init__(self.N)
        # number_of_steps = random.randint(1,2)
        number_of_steps = random.randint(1,max_steps)
        self.randomize(number_of_steps)
        return self.render()

    def checkGoal(self):
        ended = True
        reward = 1.0
        for idx in range(self.stickers.shape[0]):
            if np.min(self.stickers[idx]) != np.max(self.stickers[idx]):
                ended = False
                reward = -0.1

        return reward, ended

    def convert_action_into_rotation(self, action):
        f,l,d = self.action_to_rot[action]
        f = self.dictface[f]
        return f,l,d

    def step(self, action):

        assert action < self.actions
        f,l,d = self.convert_action_into_rotation(action)
        self.move(f,l,d)
        reward, done = self.checkGoal()
        state = self.render()
        plt.close('all')
        if reward == None:
            print(done)
            print(reward)
            return state, reward, done
        else:
            return state, reward, done


if __name__ == "__main__":
    """
    Functional testing.
    """
    np.random.seed(42)
    c = gameEnv(3)
#    c.turn("U", 1)
#    c.move("U", 0, -1)
#    swap_off_diagonal(c, "R", 2, 1)
#    c.move("U", 0, 1)
#    swap_off_diagonal(c, "R", 3, 2)
#    checkerboard(c)
    for m in range(4):
        import cv2
        cv2.imwrite("test%02d.png" % m, c.render())
        state, reward, done = c.step(1)
    cv2.imwrite("test%02d.png" % (m+1), c.render())

    #plt.close('all')