import yaml
from collections import OrderedDict
import os
import sys
from .tools import full_path

def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

def ordered_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


    # modifiled from https://github.com/ColwynGulliford/distgen/blob/master/distgen/generator.py
def parse_yaml(input):
    """
    parse yaml file to a dictionary
    :param input: yaml file path or file stream
    :return: config dictionary
    """
    if isinstance(input, str):
        if os.path.exists(full_path(input)):
            filename = full_path(input)
            with open(filename) as f:
                input_dic = ordered_load(f)
    else:
        # try if input is a stream
        try:
            input_dic = ordered_load(input)
            assert isinstance(input_dic, dict), f'ERROR: parsing unsuccessful, could not read {input}'
        except Exception as ex:
            print(ex)
            sys.exit(1)

    return input_dic
