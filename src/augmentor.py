from enum import Enum, auto
import math
import random
import copy

import numpy as np
import torchvision.transforms.functional as TF


class Transformation(object):
    def __init__(
            self, range_boundary, mutate_step,
            center_point=0, shape=None, rand_init=True):
        self.mutate_step = mutate_step

        if isinstance(range_boundary, str):
            assert shape is not None and range_boundary.endswith('%')
            range_boundary = int(float(range_boundary.rstrip('%')) / 100. * shape)

        rmin = center_point - range_boundary
        rmax = center_point + range_boundary + 1e-5
        self.range = np.arange(rmin, rmax, self.mutate_step)

        self.value = random.choice(self.range) if rand_init is True else center_point

    def reset(self):
        self.value = self.range[len(self.range) // 2]


class Chromosome(object):
    class Mutator(Enum):
        ADD  = auto()
        SUB  = auto()
        FLIP = auto()

    def __init__(self, image_shape, **kwargs):
        # rotate: [-30, 30, 1] degrees
        self.rotate = Transformation(30, 1, **kwargs)
        # translate: [-10%, 10%, 1] pixels
        self.translate = Transformation('10%', 1, shape=image_shape[0], **kwargs)
        self.translate_v = Transformation('10%', 1, shape=image_shape[1], **kwargs)
        # shear: [-10%, 10%, 3.0] degrees
        self.shear = Transformation('10%', 3., shape=180, **kwargs)
        # zoom: [0.9, 1.1, 0.05] factor
        self.zoom = Transformation(0.1, 0.05, center_point=1, **kwargs)
        # brighten: [0.8, 1.2, 0.05] factor
        self.brighten = Transformation(0.2, 0.05, center_point=1, **kwargs)
        # contrast: [0.8, 1.2, 0.05] factor
        self.contrast = Transformation(0.2, 0.05, center_point=1, **kwargs)

        self.trans = [attr for attr in dir(self) \
                if not callable(getattr(self, attr)) and not attr.startswith('__')]

    def __call__(self, x):
        x = TF.affine(
            x,
            self.rotate.value,
            [self.translate.value, self.translate_v.value],
            self.zoom.value,
            self.shear.value
        )
        x = TF.adjust_brightness(x, self.brighten.value)
        x = TF.adjust_contrast(x, self.contrast.value)
        return x

    def mutate(self):
        item = copy.deepcopy(self)
        num_choice = math.ceil(len(self.trans) / 2. - 1e-5)
        choices = random.sample(self.trans, num_choice)
        for c in choices:
            t = getattr(self, c)
            op = random.choice(list(self.Mutator))
            if op is self.Mutator.ADD:
                value = min(max(t.range[0], t.value + t.mutate_step), t.range[-1])
            elif op is self.Mutator.SUB:
                value = min(max(t.range[0], t.value - t.mutate_step), t.range[-1])
            elif op is self.Mutator.FLIP:
                value = t.range[0] + t.range[-1] - t.value
            else:
                raise NotImplementedError
            setattr(item, c, value)
        return item

    def crossover(self, other):
        item1, item2 = copy.deepcopy(self), copy.deepcopy(other)
        mask = random.choice(range(1, 2**len(self.trans)))  # 0000001 - 1111110
        ids = '{0:07b}'.format(mask)
        for c, i in zip(self.trans, ids[::-1]):  # revert
            if i:
                t1, t2 = getattr(self, c), getattr(other, c)
                setattr(item2, c, t1)
                setattr(item1, c, t2)
        return item1, item2


class GeneticAugmentor(object):
    class Fitness(Enum):
        SMALLEST = False
        LARGEST  = True

    def __init__(self, image_shape, popsize, crossover_prob):
        self.co_prob = crossover_prob
        self.popsize = popsize
        self.chromo = [Chromosome(image_shape) for _ in range(popsize)]

    def generate_population(self):
        prob = random.uniform(0, 1)
        if prob < self.co_prob:
            while len(self.chromo) < self.popsize * 2:
                item1, item2 = random.sample(self.chromo[:self.popsize], 2)
                new_item1, new_item2 = item1.crossover(item2)
                self.chromo.append(new_item1)
                self.chromo.append(new_item2)
        else:
            for i in range(self.popsize):
                new_item = self.chromo[i].mutate()
                self.chromo.append(new_item)
        return self.chromo

    def fitness(self, scores, strategy: Fitness):
        assert len(scores) == len(self.chromo)
        self.chromo = [c for _, c in sorted(
            zip(scores, self.chromo),
            key=lambda pair: pair[0],
            reverse=strategy.value
        )][:self.popsize]

    def __call__(self, x, single=True):
        if single is True:
            return [self.chromo[0](x)]
        else:
            return [c(x) for c in self.chromo]


class NeighborAugmentor(object):
    def __init__(self, image_shape, num_neighbors, seed):
        self.chromo = Chromosome(image_shape, rand_init=False)
        self.neighbors, self.accu_nhb, self.n2t = self._init_neighbors(num_neighbors, seed)

    def _init_neighbors(self, num_neighbors, seed):
        fixed_random = random.Random(seed)

        num_per_tran = { t: 0 for t in self.chromo.trans }
        tran_weights = [ len(getattr(self.chromo, t).range) for t in self.chromo.trans ]
        for _ in range(num_neighbors):
            t = fixed_random.choices(self.chromo.trans, weights=tran_weights, k=1)[0]
            num_per_tran[t] += 1

        neighbors = {}
        for t, k in num_per_tran.items():
            values = fixed_random.choices(getattr(self.chromo, t).range, k=k)
            neighbors[t] = values

        accu_neighbors = []
        sum_neighbors = 0
        for t in self.chromo.trans:
            accu_neighbors.append(sum_neighbors)
            sum_neighbors += num_per_tran[t]

        n2t = [
            i
            for i, t in enumerate(self.chromo.trans)
            for _ in range(num_per_tran[t])
        ]

        return neighbors, accu_neighbors, n2t

    def __call__(self, x, n_idx):
        t_idx = self.n2t[n_idx]
        v_idx = n_idx - self.accu_nhb[t_idx]
        tran = getattr(self.chromo, self.chromo.trans[t_idx])
        tran.value = tran.range[v_idx]
        x = self.chromo(x)
        tran.reset()
        return x

