from __future__ import absolute_import, division, print_function
from ibeis_plugin_example.version import version as __version__  # NOQA
from ibeis.control import controller_inject  # NOQA


_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_ibeis_flask_api(__name__)


@register_ibs_method
def ibeis_plugin_example_hello_world(ibs):
    args = (ibs, )
    resp = '[ibeis_plugin_example] hello world with IBEIS controller %r' % args
    return resp


@register_api('/api/plugin/example/helloworld/', methods=['GET'])
def ibeis_plugin_example_hello_world_rest(ibs):
    return ibs.ibeis_plugin_example_hello_world()
