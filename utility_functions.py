# Import standard library modules
import os
import datetime

"""
Module Name: utility_functions.py

A collection of helper functions used mainly by the CSR3D class. These functions help with filenaming
and input processing.
"""

def full_path(path):
    """
    From C. Mayes
    Helper function to expand enviromental variables and return the absolute path
    """
    return os.path.abspath(os.path.expandvars(path))

def isotime():
    """
    Return the current UTC time in ISO 8601 format with Local TimeZone information without microsecond
    """
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace(':', '_')
    #return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone().replace(microsecond=0).isoformat().replace(':','_')

def check_input_consistency(input):
        """
        Checks to make sure that the dictionary we are using for our CSR2D configuration has the correct format
        Parameters:
            input: the dictionary in question
            class_name: the name of the class (for)
        Returns:
            returns nothing if the dictionary has the correct format, if not asserts what is wrong
        """
        # TODO: need modification if dipole_config.yaml format changed
        # Must be given a beam and a lattice
        required_inputs = ['input_beam', 'input_lattice']

        allowed_params = required_inputs + ['particle_deposition', 'distribution_interpolation', 'CSR_integration',
                                                 'CSR_computation']

        # Make sure all keys in input are allowed
        for input_param in input:
            assert input_param in allowed_params, f'Incorrect param given to CSR3D.__init__(**kwargs): {input_param}\nAllowed params: {allowed_params}'

        # Make sure all required parameters are specified
        for req in required_inputs:
            assert req in input, f'Required input parameter {req} to CSR3D.__init__(**kwargs) was not found.'