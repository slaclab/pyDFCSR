def bmad_to_particlegroup_data(bmad_data):
    """
    Convert Bmad particle data to a ParticleGroup data dictionary.

    This function reverses the conversion done by particlegroup_to_bmad, mapping
    Bmad phase space coordinates back to a ParticleGroup data format.

    Parameters
    ----------
    bmad_data : dict
        A dictionary containing Bmad phase space coordinates.

    Returns
    -------
    dict
        A dictionary of data suitable for instantiating a ParticleGroup.
    """

    # Conversion to ParticleGroup units
    species = bmad_data['species']
    mc2 = mass_of(species)

    p0c = bmad_data['p0c']
    tref = bmad_data.get('tref', 0)

    p = (1 + bmad_data['pz']) * p0c
    px = bmad_data['px'] * p0c
    py = bmad_data['py'] * p0c
    pz = np.sqrt(p ** 2 - px ** 2 - py ** 2)
    gamma2 = (p / mc2) ** 2 + 1
    beta = np.sqrt(1 - 1 / gamma2)

    pg_data = {
        'x': bmad_data['x'],
        'px': px,
        'y': bmad_data['y'],
        'py': py,
        'z': np.zeros(len(p)),  # Zero by definition in z-coordinates
        'pz': pz,
        't': tref - bmad_data['z'] / (beta * c_light),
        'species': species,
        'weight': bmad_data['charge'],
        'status': bmad_data['state'],
    }

    return pg_data
