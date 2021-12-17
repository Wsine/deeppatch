from enum import Enum, auto
import math
import random
import copy

import numpy as np
import torchvision.transforms.functional as TF


class Transformation(object):
    def __init__(
            self, mutate_step, range_boundary, range_step,
            center_point=0, shape=None):
        self.mutate_step = mutate_step

        if isinstance(range_boundary, str):
            assert shape is not None and range_boundary.endswith('%')
            range_boundary = int(float(range_boundary.rstrip('%')) / 100. * shape)

        self.rmin = center_point - range_boundary
        self.rmax = center_point + range_boundary + 1e-5

        rrange = np.arange(self.rmin, self.rmax, range_step)
        self.value = random.choice(rrange)


class Chromosome(object):
    class Mutator(Enum):
        ADD  = auto()
        SUB  = auto()
        FLIP = auto()

    def __init__(self, image_shape):
        self.rotate = Transformation(6, 30, 1)  # [-30, 30, 1] degrees
        # translate: [-10%, 10%, 1] pixels
        self.translate = Transformation(1, '10%', 1, shape=image_shape[0])
        self.translate_v = Transformation(1, '10%', 1, shape=image_shape[1])
        self.shear = Transformation(3.6, '10%', 0.9, shape=180)  # [-10%, 10%, 0.9] degrees
        self.zoom = Transformation(0.02, 0.1, 0.01, center_point=1) # [0.9, 1.1, 0.001] factor
        self.brighten = Transformation(0.05, 0.2, 0.025, center_point=1) # [0.8, 1.2, 0.025] factor
        self.contrast = Transformation(0.05, 0.2, 0.025, center_point=1) # [0.8, 1.2, 0.025] factor
        self.trans = [attr for attr in dir(self)
                    if not callable(getattr(self, attr)) and not attr.startswith('__')]

    def __call__(self, x):
        x = TF.affine(
            x,
            self.rotate.value,
            [self.translate.value, self.translate_v.value],
            self.zoom.value,
            [self.shear.value, 1.]  # horizontal only
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
                value = min(max(t.rmin, t.value + t.mutate_step), t.rmax)
            elif op is self.Mutator.SUB:
                value = min(max(t.rmin, t.value - t.mutate_step), t.rmax)
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

