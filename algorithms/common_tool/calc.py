import numpy as np
import _warnings
from NYQH_NZW_QH.src.common_tool.package_tools import Exporter
import inspect
import functools
import xarray as xr
from NYQH_NZW_QH.src.common_tool.units import _mutate_arguments,setup_registry,is_quantity,process_units
import pint

exporter = Exporter(globals())
# Make our modifications using pint's application registry--which allows us to better
# interoperate with other libraries using Pint.
units = setup_registry(pint.get_application_registry())

def preprocess_and_wrap(broadcast=None, wrap_like=None, match_unit=False, to_magnitude=False):
    """Return decorator to wrap array calculations for type flexibility.

    Assuming you have a calculation that works internally with `pint.Quantity` or
    `numpy.ndarray`, this will wrap the function to be able to handle `xarray.DataArray` and
    `pint.Quantity` as well (assuming appropriate match to one of the input arguments).

    Parameters
    ----------
    broadcast : Sequence[str] or None
        Iterable of string labels for arguments to broadcast against each other using xarray,
        assuming they are supplied as `xarray.DataArray`. No automatic broadcasting will occur
        with default of None.
    wrap_like : str or array-like or tuple of str or tuple of array-like or None
        Wrap the calculation output following a particular input argument (if str) or data
        object (if array-like). If tuple, will assume output is in the form of a tuple,
        and wrap iteratively according to the str or array-like contained within. If None,
        will not wrap output.
    match_unit : bool
        If true, force the unit of the final output to be that of wrapping object (as
        determined by wrap_like), no matter the original calculation output. Defaults to
        False.
    to_magnitude : bool
        If true, downcast xarray and Pint arguments to their magnitude. If false, downcast
        xarray arguments to Quantity, and do not change other array-like arguments.
    """
    def decorator(func):
        sig = inspect.signature(func)
        if broadcast is not None:
            for arg_name in broadcast:
                if arg_name not in sig.parameters:
                    raise ValueError(
                        f'Cannot broadcast argument {arg_name} as it is not in function '
                        'signature'
                    )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind(*args, **kwargs)

            # Auto-broadcast select xarray arguments, and update bound_args
            if broadcast is not None:
                arg_names_to_broadcast = tuple(
                    arg_name for arg_name in broadcast
                    if arg_name in bound_args.arguments
                    and isinstance(
                        bound_args.arguments[arg_name],
                        (xr.DataArray, xr.Variable)
                    )
                )
                broadcasted_args = xr.broadcast(
                    *(bound_args.arguments[arg_name] for arg_name in arg_names_to_broadcast)
                )
                for i, arg_name in enumerate(arg_names_to_broadcast):
                    bound_args.arguments[arg_name] = broadcasted_args[i]

            # Cast all Variables to their data and warn
            # (need to do before match finding, since we don't want to rewrap as Variable)
            def cast_variables(arg, arg_name):
                _warnings.warn(
                    f'Argument {arg_name} given as xarray Variable...casting to its data. '
                    'xarray DataArrays are recommended instead.'
                )
                return arg.data
            _mutate_arguments(bound_args, xr.Variable, cast_variables)

            # Obtain proper match if referencing an input
            match = list(wrap_like) if isinstance(wrap_like, tuple) else wrap_like
            if isinstance(wrap_like, str):
                match = bound_args.arguments[wrap_like]
            elif isinstance(wrap_like, tuple):
                for i, arg in enumerate(wrap_like):
                    if isinstance(arg, str):
                        match[i] = bound_args.arguments[arg]

            # Cast all DataArrays to Pint Quantities
            _mutate_arguments(bound_args, xr.DataArray, lambda arg, _: arg.metpy.unit_array)

            # Optionally cast all Quantities to their magnitudes
            if to_magnitude:
                _mutate_arguments(bound_args, units.Quantity, lambda arg, _: arg.m)

            # Evaluate inner calculation
            result = func(*bound_args.args, **bound_args.kwargs)

            # Wrap output based on match and match_unit
            if match is None:
                return result
            else:
                if match_unit:
                    wrapping = _wrap_output_like_matching_units
                else:
                    wrapping = _wrap_output_like_not_matching_units

                if isinstance(match, list):
                    return tuple(wrapping(*args) for args in zip(result, match))
                else:
                    return wrapping(result, match)
        return wrapper
    return decorator

def _wrap_output_like_matching_units(result, match):
    """Convert result to be like match with matching units for output wrapper."""
    output_xarray = isinstance(match, xr.DataArray)
    match_units = str(match.metpy.units if output_xarray else getattr(match, 'units', ''))

    if isinstance(result, xr.DataArray):
        result = result.metpy.convert_units(match_units)
        return result if output_xarray else result.metpy.unit_array
    else:
        result = (
            result.to(match_units) if is_quantity(result)
            else units.Quantity(result, match_units)
        )
        return (
            xr.DataArray(result, coords=match.coords, dims=match.dims) if output_xarray
            else result
        )

def _wrap_output_like_not_matching_units(result, match):
    """Convert result to be like match without matching units for output wrapper."""
    output_xarray = isinstance(match, xr.DataArray)
    if isinstance(result, xr.DataArray):
        return result if output_xarray else result.metpy.unit_array
    # Determine if need to upcast to Quantity
    if (
        not is_quantity(result) and (
            is_quantity(match) or (output_xarray and is_quantity(match.data))
        )
    ):
        result = units.Quantity(result)
    return (
        xr.DataArray(result, coords=match.coords, dims=match.dims)
        if output_xarray and result is not None
        else result
    )

def dewpoint_from_relative_humidity(temperature, relative_humidity):
    r"""Calculate the ambient dewpoint given air temperature and relative humidity.

    Parameters
    ----------
    temperature : `pint.Quantity`
        Air temperature

    relative_humidity : `pint.Quantity`
        Relative humidity expressed as a ratio in the range 0 < relative_humidity <= 1

    Returns
    -------
    `pint.Quantity`
        Dewpoint temperature

    Examples
    --------
    >>> from metpy.calc import dewpoint_from_relative_humidity
    >>> from metpy.units import units
    >>> dewpoint_from_relative_humidity(10 * units.degC, 50 * units.percent)
    <Quantity(0.0536760815, 'degree_Celsius')>

    .. versionchanged:: 1.0
       Renamed ``rh`` parameter to ``relative_humidity``

    See Also
    --------
    dewpoint, saturation_vapor_pressure

    """
    if np.any(relative_humidity > 1.2):
        _warnings.warn('Relative humidity >120%, ensure proper units.')
    return dewpoint(relative_humidity * saturation_vapor_pressure(temperature))

@exporter.export
@preprocess_and_wrap(wrap_like='vapor_pressure')
@process_units({'vapor_pressure': '[pressure]'}, '[temperature]', output_to=units.degC)
def dewpoint(vapor_pressure):
    r"""Calculate the ambient dewpoint given the vapor pressure.

    Parameters
    ----------
    vapor_pressure : `pint.Quantity`
        Water vapor partial pressure

    Returns
    -------
    `pint.Quantity`
        Dewpoint temperature

    Examples
    --------
    >>> from metpy.calc import dewpoint
    >>> from metpy.units import units
    >>> dewpoint(22 * units.hPa)
    <Quantity(19.0291018, 'degree_Celsius')>

    See Also
    --------
    dewpoint_from_relative_humidity, saturation_vapor_pressure, vapor_pressure

    Notes
    -----
    This function inverts the [Bolton1980]_ formula for saturation vapor
    pressure to instead calculate the temperature. This yields the following formula for
    dewpoint in degrees Celsius, where :math:`e` is the ambient vapor pressure in millibars:

    .. math:: T = \frac{243.5 \log(e / 6.112)}{17.67 - \log(e / 6.112)}

    .. versionchanged:: 1.0
       Renamed ``e`` parameter to ``vapor_pressure``

    """
    val = np.log(vapor_pressure / units.Quantity(6.112, 'millibar').m_as('Pa'))
    return units.Quantity(0., 'degC').m_as('K') + 243.5 * val / (17.67 - val)

@exporter.export
@preprocess_and_wrap(wrap_like='temperature')
@process_units({'temperature': '[temperature]'}, '[pressure]')
def saturation_vapor_pressure(temperature):
    r"""Calculate the saturation water vapor (partial) pressure.

    Parameters
    ----------
    temperature : `pint.Quantity`
        Air temperature

    Returns
    -------
    `pint.Quantity`
        Saturation water vapor (partial) pressure

    Examples
    --------
    >>> from metpy.calc import saturation_vapor_pressure
    >>> from metpy.units import units
    >>> saturation_vapor_pressure(25 * units.degC).to('hPa')
    <Quantity(31.6742944, 'hectopascal')>

    See Also
    --------
    vapor_pressure, dewpoint

    Notes
    -----
    Instead of temperature, dewpoint may be used in order to calculate
    the actual (ambient) water vapor (partial) pressure.

    The formula used is that from [Bolton1980]_ for T in degrees Celsius:

    .. math:: 6.112 e^\frac{17.67T}{T + 243.5}

    """
    # Converted from original in terms of C to use kelvin.
    return units.Quantity(6.112, 'millibar').m_as('Pa') * np.exp(
        17.67 * (temperature - 273.15) / (temperature - 29.65)
    )
