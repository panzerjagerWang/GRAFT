"""Training-time augmentations, inserted into the Ultralytics data pipeline before mosaic.

install(kind, args, seed) patches YOLODataset.get_image_and_label so that every training image (also the extra tiles of a mosaic)
passes through the augmentation; validation and prediction are untouched.  An augmentation is called as aug(label, rng, dataset)
with the label dict of one image (the BGR image resized to a long side of 640 px, normalized xywh boxes in label['instances'],
classes in label['cls']) and returns it changed.  It must not change the image array in place, because Ultralytics keeps it in a
buffer.  kind may chain several augmentations with '+'; args then holds shared keys plus one dict per part, e.g.
{"bank": ..., "graft": {"relight": "none"}, "snow": {"p": 0.3}}.
"""
import numpy as np

REGISTRY = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        return cls
    return deco


def build(kind, args):
    import aug_methods  # noqa: F401  (registers the augmentations)
    parts = kind.split('+')
    shared = {k: v for k, v in args.items() if k not in parts}
    objs = [REGISTRY[k](**{**shared, **args.get(k, {})}) for k in parts]
    return objs[0] if len(objs) == 1 else aug_methods.Chain(objs)


def install(kind, args, seed=0):
    if kind == 'base':
        return None
    from ultralytics.data.dataset import YOLODataset
    aug = build(kind, args)
    orig = YOLODataset.get_image_and_label
    state = {'rng': None}

    def get_image_and_label(self, index):
        label = orig(self, index)
        if not self.augment:
            return label
        if state['rng'] is None:  # one generator per dataloader worker
            import torch
            info = torch.utils.data.get_worker_info()
            state['rng'] = np.random.default_rng([seed, 0 if info is None else info.id + 1, int(np.random.randint(1 << 30))])
        return aug(label, state['rng'], self)

    YOLODataset.get_image_and_label = get_image_and_label
    return aug
