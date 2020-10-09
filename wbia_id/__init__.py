# -*- coding: utf-8 -*-
from wbia_id import _plugin  # NOQA

try:
    from wbia_id._version import __version__  # NOQA
except ImportError:
    __version__ = '0.0.0'
