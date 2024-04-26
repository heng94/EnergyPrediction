from .registry import Registry
from .utils import check_availability


MODEL_REGISTRY = Registry("MODEL")


def build_model(args):
    avai_models = MODEL_REGISTRY.registered_names()
    check_availability(args.model.name, avai_models)
    print("Loading model: {}".format(args.model.name))
    return MODEL_REGISTRY.get(args.model.name)(args)
