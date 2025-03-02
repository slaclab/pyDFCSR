# Import standard library modules
import os
import sys
from collections import OrderedDict

# Import third-party modules
import yaml

# Import modules specific to this package
from utility_functions import full_path

def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    """
    Parses a stream to a dictionary, uses safe loader by default
    parameters - stream: the stream to be parsed
    :return: configured dictionary
    """

    # Define subclass of 'Loader' class
    class OrderedLoader(Loader):
        pass
    
    # Define a custom constructor for ordered mappings and add it to our subclass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    
    # Parse the stream via our custom constructor
    return yaml.load(stream, OrderedLoader)

def ordered_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    """
    Serializes dictionary into YAML
    parameters - data: dictionary, stream: optional location to dump yaml representation
    :return: YAML representation
    """
    # Define subclass of 'Dumper' class
    class OrderedDumper(Dumper):
        pass

    # Define a callback function to handle the representation of 'OrderedDict' instances
    # when handling serializing YAML
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    
    OrderedDumper.add_representer(OrderedDict, _dict_representer)

    # Return the yaml representation from data serialized using our ordered dumper
    return yaml.dump(data, stream, OrderedDumper, **kwds)


    # modifiled from https://github.com/ColwynGulliford/distgen/blob/master/distgen/generator.py
def parse_yaml(input):
    """
    Parse yaml file to a dictionary
    parameters - input: yaml file path or file stream
    :return: config dictionary
    """

    # If our input type is string, then we assume that the string is a file path and we 
    # attempt to open it
    if isinstance(input, str):
        if os.path.exists(full_path(input)):
            filename = full_path(input)

            # Open the file and create dictionary using ordered_load() 
            with open(filename) as f:
                input_dic = ordered_load(f)

    # If our input type is not a string, then we attempt to parse it as a stream
    else:
        # try if input is a stream
        try:
            input_dic = ordered_load(input)
            assert isinstance(input_dic, dict), f'ERROR: parsing unsuccessful, could not read {input}'
        except Exception as ex:
            print(ex)
            sys.exit(1)

    return input_dic
