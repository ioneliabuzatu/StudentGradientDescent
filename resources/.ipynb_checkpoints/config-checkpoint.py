"""
Utility classes for working with configuration files.

Copyright statement:
This  material,  no  matter  whether  in  printed  or  electronic  form,
may be  used  for personal  and non-commercial educational use only.
Any reproduction of this manuscript, no matter whether as a whole or in parts,
no matter whether in printed or in electronic form,
requires explicit prior acceptance of the authors.

"""

__version__ = "0.2"

__author__ = "Pieter-Jan Hoedt"
__email__ = "hoedt@ml.jku.at"
__date__ = "01-03-2021"
__copyright__ = "Copyright 2019-2021, JKU Institute for Machine Learning"


from collections.abc import Mapping
from pathlib import Path
from typing import Iterator


class Configuration(Mapping):
    """
    Mapping of hyper-parameters to their corresponding values.

    `Configuration` provides a convenient way to store and access
     hyper-parameters for machine learning experiments.
     It provides a dict-like interface to the configuration
     and allows for arbitrary hierarchies of configurations.
     Additionally, it exposes hyper-parameters and sub-configurations
     as attributes to save you from typing brackets and strings everywhere.

     Examples
     --------
     Configurations behave very much like dicts

     >>> conf = Configuration(epochs=5, optimiser={'name': 'sgd', 'lr': 1e-3})
     >>> conf
     {'epochs': 5, 'optimiser': {'name': 'sgd', 'lr': 0.001}}
     >>> conf['epochs']
     5

     Entries can also be accessed as attributes

     >>> conf.epochs
     5
     >>> conf.optimiser.lr
     1e-3

     New parameters can be added

     >>> conf.batch_size = 5
     >>> conf.optimiser.weight_decay = .1

     Existing parameters must be overwritten explicitly

     >>> conf.epochs = 10
     Traceback: ... TypeError
     >>> conf.overwrite('epochs', 10)
     5

     Pass configurations as keyword arguments to functions

     >>> def some_func(**kwargs):
     ...     print(kwargs)
     >>> some_func(**conf.optimiser)
     {'name': 'sgd', 'lr': 0.001, 'weight_decay': 5}
    """

    def __init__(self, **config):
        self._config = {}

        for k, v in config.items():
            self[k] = v

    def __getitem__(self, k: str) -> object:
        return self._config[k]

    def __setitem__(self, k: str, val: object):
        if not k[0].isalpha():
            raise ValueError("configuration keys must start with letter")
        elif k in self._config:
            msg = f"'{self.__class__.__name__}' object does not allow "
            msg += f"implicit overwriting of entries, "
            raise TypeError(msg + "use `Configuration.overwrite` instead")

        try:
            # convert mappings to sub-configs
            val = Configuration(**val)
        except TypeError:
            pass
        finally:
            self._config[k] = val

    def __delitem__(self, k: str):
        del self._config[k]

    def __len__(self) -> int:
        return self._config.__len__()

    def __iter__(self) -> Iterator[object]:
        return self._config.__iter__()

    def __repr__(self):
        return self._config.__repr__()

    def __getattr__(self, k: str) -> object:
        if k.startswith('_'):
            raise AttributeError(f"no attribute '{k}'")

        try:
            return self.__getitem__(k)
        except KeyError as e:
            raise AttributeError(f"no config entry with key '{k}'") from e

    def __setattr__(self, k: str, val: object):
        if k.startswith('_'):
            return super().__setattr__(k, val)

        return self.__setitem__(k, val)

    def __delattr__(self, k: str):
        if k.startswith('_'):
            return super().__delattr__(k)

        try:
            return self.__delitem__(k)
        except KeyError as e:
            raise AttributeError(f"no config entry with key '{k}'") from e

    def overwrite(self, k: str, val: object) -> object:
        """
        Explicitly overwrite the value for a hyper-parameter.

        Parameters
        ----------
        k : str
            Name (or key) for the hyper-parameter to overwrite.
        val : object
            The new value for the hyper-parameter specified by `k`.
            Must have the same type as the current value.

        Returns
        -------
        old_val : object
            The old value for the hyper-parameter specified by `k`.

        """
        old_val = self.__getitem__(k)
        if type(old_val) != type(val):
            msg = "new value must be of the same type as the old value, "
            msg += f"but got '{type(val)}' instead of '{type(old_val)}'"
            raise ValueError(msg)

        self.__delitem__(k)
        self.__setitem__(k, val)
        return old_val

    @property
    def dict(self) -> dict:
        """ This configuration object as plain dictionary. """
        return {k: v.dict if isinstance(v, Configuration) else v
                for k, v in self._config.items()}


class Grid:
    """
    Grid of hyper-parameter configurations.

    `Grid` provides an easy  way to set up
    a grid of hyper-parameter configurations.
    It collects a list of possible values
    for each individual hyper-parameter.
    Iterating over the grid will result
    in a sequence of all possible configurations.

    Examples
    --------
    Create a grid with a single configuration and add options as necessary
    >>> sgd_grid = Grid(name='SGD')
    >>> sgd_grid.add_options('lr', [1e-3, 1e-2, 1e-4])

    Hyper-parameters can be grids again:
    >>> grid = Grid(epochs=5, opt=sgd_grid)
    >>> grid.add_option('opt', Grid(name='adam', lr=1e-3))

    Iteration over grid gives all possible configs as `Configuration`:
    >>> for conf in grid:
    ...     print(conf)
    {epochs: 5, opt: {lr: 0.001, name: SGD}}
    {epochs: 5, opt: {lr: 0.01, name: SGD}}
    {epochs: 5, opt: {lr: 0.0001, name: SGD}}
    {epochs: 5, opt: {lr: 0.001, name: adam}}

    Indexing allows random sampling from grid:
    >>> rng = np.random.RandomState(1806)
    >>> for i in rng.choice(len(grid), 2, replace=False)
    >>>     print(i, grid[i])
    {epochs: 5, opt: {lr: 0.001, name: adam}}
    {epochs: 5, opt: {lr: 0.0001, name: SGD}}
    """

    def __init__(self, **kwargs):
        self._options = {}

        for k, v in kwargs.items():
            self.add_option(k, v)

    def __getitem__(self, index: int) -> Configuration:
        if not self._options:
            raise IndexError

        res = {}
        for key, options in self._options.items():
            if isinstance(options[0], Grid):
                grid_lengths = [len(g) for g in options]
                index, opt_index = divmod(index, sum(grid_lengths))

                # find grid and index in that grid
                for i, grid_len in enumerate(grid_lengths):
                    if opt_index < grid_len:
                        break
                    opt_index -= grid_len

                options = options[i]
            else:
                index, opt_index = divmod(index, len(options))

            res[key] = options[opt_index]

        return Configuration(**res)

    def __len__(self) -> int:
        if not self._options:
            return 0

        prod = 1
        for options in self._options.values():
            if isinstance(options[0], Grid):
                prod *= sum(len(g) for g in options)
            else:
                prod *= len(options)
        return prod

    def __iter__(self) -> iter:
        def _cart_prod(_dict):
            """ Compute Cartesian product of options. """
            if len(_dict) == 0:
                yield {}
                return

            # recursion over different parameters
            k, options = _dict.popitem()
            for res in _cart_prod(_dict):
                for opt in options:
                    _grid = opt if isinstance(opt, Grid) else [opt]
                    for c in _grid:
                        yield dict(**res, **{k: c})

        # convert dicts to proper configurations
        for conf in _cart_prod(dict(self._options)):
            yield Configuration(**conf)

    def get_options(self, k: str):
        return self._options[k]

    def add_option(self, k: str, val: object):
        self.add_options(k, [val])

    def add_options(self, k: str, vals: iter):
        arr = self._options.setdefault(k, [])
        arr.extend(vals)


def _config_path(path: str = None, ext: str = ".yml") -> Path:
    """ Builds the default path to the config file if necessary. """
    file_path = Path() if path is None else Path(path)
    if file_path.is_dir():
        file_path = file_path / "config"

    if not file_path.suffix:
        file_path = file_path.with_suffix(ext)

    return file_path


def read_config(path: str = None, yaml: bool = True) -> Configuration:
    """
    Read a configuration from a file.

    Parameters
    ----------
    path : str
        Path to the configuration file or to a directory that contains
        a file with the name "config.yaml" or "config.json".
    yaml : bool, optional
        If `True`, the file is assumed to be in the YAML format (the default).
        Otherwise, the file is expected to be in the JSON format.

    Returns
    -------
    config: Configuration
        A configuration with the parameters as specified by the file.
    """
    extension = ".yaml" if yaml else ".json"
    file = _config_path(path, extension)
    if extension == ".json":
        from json import load
    elif extension == ".yaml":
        from yaml import safe_load as load
    else:
        raise ValueError(f"unknown config file extension: '{extension}'")

    with open(file, 'r') as fp:
        data = load(fp)

    return Configuration(**data)


def write_config(config: Configuration, path: Path = None, yaml: bool = True) -> Path:
    """
    Write a configuration to a file.

    Parameters
    ----------
    config : Configuration
        The configuration to write to a file.
    path : str
        Full path for where to store the configuration file or
        path to a directory where the file will be stored as "config.yaml/json".
    yaml : bool, optional
        If `True`, the file is assumed to be in the YAML format (the default).
        Otherwise, the file is expected to be in the JSON format.

    Returns
    -------
    file : Path
        Path to the file with the contents of the configuration.
    """
    extension = ".yaml" if yaml else ".json"
    file = _config_path(path, extension)
    if extension == ".json":
        from json import dump as _dump

        def dump(o, f):
            return _dump(o, f, indent='\t')
    elif extension == ".yaml":
        from yaml import safe_dump as dump
    else:
        raise ValueError(f"unknown config file extension: '{extension}'")

    with open(file, 'w') as fp:
        dump(config.dict, fp)

    return file
