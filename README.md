# IBEIS Plugin Example
An example of how to design and use a Python module as a plugin in the IBEIS IA system

# Installation

Install this plugin as a Python module using

```bash
cd ~/code/ibeis_plugin_example/
python setup.py develop
```

# REST API

With the plugin installed, register the module name with the `IBEISControl.py` file
in the ibeis repository located at `ibeis/ibeis/control/IBEISControl.py`.  Register
the module by adding the string (for example, `ibeis_plugin_example`) to the
list `AUTOLOAD_PLUGIN_MODNAMES`.

Then, load the web-based IBEIS IA service and open the URL that is registered with
the `@register_api decorator`.

```bash
cd ~/code/ibeis/
python dev.py --web
```

Navigate in a browser to http://127.0.0.1:5000/api/plugin/example/helloworld/ where
this returns a formatted JSON response, including the serialized returned value
from the `ibeis_plugin_example_hello_world()` function

```
{"status": {"cache": -1, "message": "", "code": 200, "success": true}, "response": "[ibeis_plugin_example] hello world with IBEIS controller <IBEISController(testdb1) at 0x11e776e90>"}
```

# Python API

```bash
python

Python 2.7.14 (default, Sep 27 2017, 12:15:00)
[GCC 4.2.1 Compatible Apple LLVM 9.0.0 (clang-900.0.37)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import ibeis
>>> ibs = ibeis.opendb()

[ibs.__init__] new IBEISController
[ibs._init_dirs] ibs.dbdir = u'/Datasets/testdb1'
[depc] Initialize ANNOTATIONS depcache in u'/Datasets/testdb1/_ibsdb/_ibeis_cache'
[depc] Initialize IMAGES depcache in u'/Datasets/testdb1/_ibsdb/_ibeis_cache'
[ibs.__init__] END new IBEISController

>>> ibs.ibeis_plugin_example_hello_world()
'[ibeis_plugin_example] hello world with IBEIS controller <IBEISController(testdb1) at 0x10b24c9d0>'
```

The function from the plugin is automatically added as a method to the ibs object
as `ibs.ibeis_plugin_example_hello_world()`, which is registered using the
`@register_ibs_method decorator`.




