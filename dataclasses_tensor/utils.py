import inspect
import sys

from typing import Any, List, Optional, Union

def _get_type_cons(type_):
    if sys.version_info.minor == 6:
        try:
            cons = type_.__extra__
        except AttributeError:
            try:
                cons = type_.__origin__
            except AttributeError:
                cons = type_
            else:
                cons = type_ if cons is None else cons
        else:
            try:
                cons = type_.__origin__ if cons is None else cons
            except AttributeError:
                cons = type_
    else:
        cons = type_.__origin__
    return cons

def _get_type_origin(type_):
    try:
        origin = type_.__origin__
    except AttributeError:
        if sys.version_info.minor == 6:
            try:
                origin = type_.__extra__
            except AttributeError:
                origin = type_
            else:
                origin = type_ if origin is None else origin
        else:
            origin = type_
    return origin

def _hasargs(type_, *args):
    try:
        return all(arg in type_.__args__ for arg in args)
    except AttributeError:
        return False

def _isinstance_safe(o, t):
    try:
        return isinstance(o, t)
    except Exception:
        return False

def _issubclass_safe(cls, classinfo):
    try:
        return issubclass(cls, classinfo)
    except Exception:
        return _is_new_type(cls) and _is_new_type_subclass_safe(cls, classinfo)

def _is_new_type_subclass_safe(cls, classinfo):
    super_type = getattr(cls, "__supertype__", None)
    if super_type:
        return _is_new_type_subclass_safe(super_type, classinfo)
    try:
        return issubclass(cls, classinfo)
    except Exception:
        return False

def _is_new_type(type_):
    return inspect.isfunction(type_) and hasattr(type_, "__supertype__")

def _is_optional(type_):
    return (_issubclass_safe(type_, Optional) or
            _hasargs(type_, type(None)) or
            type_ is Any)

def _is_list(type_):
    return _issubclass_safe(type_, List) or _get_type_origin(type_) is list

def _is_union(type_):
    return _get_type_origin(type_) is Union
