from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject
from ibeis.constants import IMAGE_TABLE, ANNOTATION_TABLE
from ibeis.constants import CONTAINERIZED, PRODUCTION  # NOQA
import numpy as np
import dtool as dt
import utool as ut
import vtool as vt
import ibeis
import tqdm
import os
(print, rrr, profile) = ut.inject2(__name__)


# IBEIS Controller and Python API
# -------------------------------
#
# IBEIS supports dynamically-injected methods to the controller object.  Any
# function that you wrap with the @register_ibs_method decorator is automatically
# added to the controller as ibs.<function_name>() method.  The function then becomes
# accessible anywhere in the codebase if one has access to the IBEIS controller.
# For example,
#
#     @register_ibs_method
#     def new_plugin_function(ibs, parameter1, *args, **kwargs):
#         return '%s in %s' (parameter1, ibs.dbname, )
#
# will be accessible at ibs.new_plugin_function('test') and will return the value
# 'test for testdb_identification' when executed if the database's containing
# folder is named "testdb_identification".
#
# Database Structure
# ------------------
# An IBEIS database, at it's core, is simply a folder on the local file system.
# IBEIS uses SQLite3 and static folders for all of its database and asset storage
# and has the following structure:
#
#     ibeis_database_folder/
#         _ibsdb/
#             _ibeis_backups/
#             _ibeis_cache/
#             _ibeis_logs/
#             images/
#             _ibeis_database.sqlite3
#             _ibeis_staging.sqlite3
#             <other miscellaneous folders and files>
#         smart_patrol/
#         <other miscellaneous folders and files>
#
# The volatile (i.e. cannot be re-computed) database files and folders are:
#
#     _ibeis_database.sqlite3
#         The primary IBEIS database.  This database contains many tables to store
#         image, annotation, name, asset properties and store various decisions and
#         computed results.
#
#     _ibeis_staging.sqlite3
#         A staging database for semi-temporary results, content in this database is
#         intended to eventually be committed or resolved into the primary database
#
#     images/
#         A folder containing all of the localized images for the datbase.  Images
#         are renamed when copied to this folder to be <UUID>.<EXT>.  The image's
#         UUID is computed deterministically based on the pixel content of the image
#         and we inherit any original file extensions to prevent loosing any metadata.
#
# We also have extensive caching processes for different computer vision and machine
# learning results along with modified versions of the original assets.  All cached
# data is stored in the _ibsdb/_ibeis_cache/ folder.  This folder can be deleted
# freely at any point to delete any pre-computed results.  Conversely, we also take
# daily snapshots of the primary and staging database whenever the IBEIS controller
# is first started.  These backups are stored in _ibsdb/_ibeis_backups/.  The IBEIS
# controller also supports incremental updates, so a database backup is also
# performed before any database update.
#
# We provide an example below on how to write a customized database controller
# along with example incremental update functions.
#
# First-order Data Objects
# ------------------------
# There are 4 main data constructs in IBEIS:
#
#     ImageSets   - A collection of images into a single group.  This grouping is
#                   used as a very general association and can indicate, for example,
#                   the set of images taken at the same time and place or the images
#                   that all contain a target species or the images that are to
#                   be used by a machine learning algorithm for training or testing.
#     Images      - Original images provided by the user or contributor.
#                   ImageSets and Images have a many-to-many relationship as
#                   more than one image can be in an ImageSet and an image can be
#                   a member of multiple ImageSets.
#     Annotations - Pixel regions within an Image to designate a single animal.
#                   Annotations are, in their most basic form, a bounding box.
#                   Bounding boxes in IBEIS are parameterized by (xtl, ytl, w, h)
#                   where "xtl" designates "x pixel coordinate for the top-left corner",
#                   "ytl" designates "y pixel coordinate for the top-left corner",
#                   "w" designates "the width of the box in pixel coordinates", and
#                   "h" designates "the height of the box in pixel coordinates".
#                   Images and Annotations have a one-to-many relationship.
#     Names       - A ID label for an annotation.  A one-to-many relationship
#                   between Names and Annotations is usually the end-result of
#                   using the IBEIS system.
#
# In general, a single instance of the IBEIS code base only has one IBEIS controller
# (commonly referred to by the variable name "ibs") and is the primary development
# handle for any new features.  The controller is packed with very handy functions,
# including:
#
#     gid_list = ibs.get_valid_gids()  # returns the internal rowids for images
#     aid_list = ibs.get_valid_aids()  # returns the internal rowids for annotations
#     nid_list = ibs.get_valid_nids()  # returns the internal rowids for names
#
# These lists of internal rowids allow for a wide range of adders, getters, setters,
# deleters and algorithm calls.  For example,
#
#     image_gps_list  = ibs.get_image_gps(gid_list)    # Returns a parallel list of image 2-tuple (lat, lon) GPS coordinates
# or
#     annot_uuid_list = ibs.get_annot_uuids(aid_list)  # Returns a parallel list of annotation UUIDs
# or
#     aids_list       = ibs.get_name_aids(nid_list)    # Returns a parallel list of lists of aids
#
# In general, if you think a function should exist to associate two object types
# or compute some value, it probably already exists.
#
# Controller API Injection
# ------------------------
# Any function that is decorated by @register_ibs_method should accept the IBEIS
# "ibs" controller object as the first parameter.
#
# IMPORTANT: To be enabled, a plug-in must be manually registered with the IBEIS
# controller.  The registration process involves adding the module import name for
# this _plugin.py file to a list of plugins.  This list lives in the main IBEIS
# repository at:
#
#     ibeis/ibeis/control/IBEISControl.py
#
# and added to the list with variable name "AUTOLOAD_PLUGIN_MODNAMES" at the top
# of the file.  On load, the IBEIS controller will add its own injected controller
# functions and will then proceed to add any external plug-in functions.  The
# injection code will look primarily for any functions with the @register_ibs_method
# decorator, but will also add any of the decorators described below for web
# requests.
_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)


# IBEIS REST API, Web Interface, and Job Engine
# ---------------------------------------------
#
# IBEIS supports a web-based REST API that allows for local functions to be called
# from public end-points.  The POST and GET arguments that are passed to the
# web server by the client are automatically parsed into Python-valid objects
# using a customized JSON converter.  Any responses from API calls are also wrapped
# by a JSON encoding and thus also need to be serialize-able.
#
# Any function that you with to expose to the web simply needs a @register_api
# decorator added above the function's definition (after any @register_ibs_method)
# decorators.  This decorator takes in the endpoint and the allowed HTTP method(s)
# (e.g. GET, POST, PUT, DELETE) that are allowed.  The same REST endpoint can point
# to different functions through the specifications of different methods.
#
# IBEIS REST API
# --------------
# There already exists an extensive REST API that mirrors the existing Python
# API, but is a curated subset of useful functions.  For example, the Python API
# function
#
#     gps_list = ibs.get_image_gps(gid_list=[1,2,3] )
#
# has a mirrored REST API interface at the endpoint
#
#     [GET] /api/image/gps/?gid_list=[1,2,3]
#
# and returns a JSON-formatted response with the same contents for gps_list.
#
# Alternatively, a user can opt to expose an endpoint that does not apply any
# JSON response serialization.  Specific examples of this need include wanting to
# serve HTML pages, raw image bytes, CSV or XML downloads, redirects or HTTP errors
# or any other non-JSON response.  Functions that are decorated with @register_route
# also benefit from the same parameter JSON serialization as @register_api.
#
# IBEIS Web Interface
# -------------------
# IBEIS also supports a basic web interface with viewing tools and curation tools.
# For example, the routes /view/imagesets/, /view/images/, /view/annotations/,
# and /view/parts/ allow for a user to quickly see the state of the database.
# There is also a batched uploading function to easily add a handful of images to
# a database without needing to use the full Python or REST APIs for larger imports.
#
# The web-tools for turking (borrowing the phrase from "Amazon Mechanical Turk")
# support a wide range of helpful functions.  The most helpful interface is
# arguably the annotation bounding box and metadata interface.  This interface
# can be viewed at /turk/detection/?gid=X and shows the current annotations for
# Image RowID=X, where they are located, what metadata is set on them, what parts
# (if any) have been added to the annotations, and what metadata is on the parts.
#
# While not required, we STRONGLY suggest all API endpoints in a plug-in to be
# prefixed with "/api/plugin/<Plug-in Name>/.../.../" to keep the registered APIs from
# conflicting.  We also suggest doing the same with your function names as well
# that are injected into the IBEIS controller on load, for example using a
# function name of ibs.ibeis_plugin_<Plug-in Name>_..._...().
#
# Background API Job Engine
# -------------------------
# The REST API has a background "job engine".  The purpose of the job engine
# is to preserve the special properties of a responsive and state-less web
# paradigm.  In short, the IBEIS web controller will automatically serialized
# concurrent requests but some API calls involve a very long processing delay.
# If a long API call is called in the main thread, it will deny any and all
# simultaneous web calls from resolving.  The solution is to mode any long API
# calls that are exposed by the web interface to the job engine to promote
# responsiveness and reduce or eliminate client-side web time outs.  Any API that
# uses the job engine (as shown in an example below) should return a job UUID
# and should be prefixed with "/api/engine/plugin/<Plug-in Name>/.../.../" to
# differentiate it from an instantaneous call.  We also recommend having any
# engine calls accept the POST HTTP method by default.
#
# A user or application can query on the status of a background job, get its
# original request metadata, and retrieve any results.  The job engine supports
# automatic callbacks with a specified URL and HTTP method when the job is
# complete.
register_api   = controller_inject.get_ibeis_flask_api(__name__)
register_route = controller_inject.get_ibeis_flask_route(__name__)


# IBEIS Dependency Graph
# ----------------------
#
# IBEIS supports a dependency-graph cache computation structure for easy pipelining.
# This is similar to other frameworks (e.g. Luigi) but has been customized and
# implemented by hand to resolve assets consistent with the IBEIS database.
#
# In general, a dependency cache (referred to by "depc" in the code base) is a
# coding tool to eliminate the complexity of managing state and staged pipeline
# configurations.  For example, say we want to compute the sum and product of a
# cryptographically secure hash of an image's pixel data.  We may want to build
# a dependency graph that looks like this
#
#                               +-----------+
#                               |           |
#                               |   Images  |
#                               |           |
#                               +-----+-----+
#                                     |
#                   +-----------------+-----------------+
#                   |                 |                 |
#             +-----+-----+     +-----+-----+     +-----+-----+
#             |           |     |           |     |           |
#             |  Feature  |     |    Hash   |     | Thumbnail |
#             |           |     |           |     |           |
#             +-----+-----+     +-----+-----+     +-----+-----+
#                                     |
#                                     +-----------------+
#                                     |                 |
#                               +-----+-----+     +-----+-----+
#                               |           |     |           |
#                               |  HashSum  |     |  HashProd |
#                               |           |     |           |
#                               +-----------+     +-----------+
#
# To compute the HashSum on an Image, we first need to compute a Hash on the
# Image.  The dependency cache will preserve this ordering by computing and
# storing intermediate results for use by the depc node you want to retrieve.
# Therefore, e would expect that any call to HashSum for Image RowID=10 would
# also compute the Hash for Image RowID=10.  Subsequently calling the call for
# HashProd on Image RowID=10 (with the same configuration) will simply retrieve
# the pre-computed Hash for RowID=10.
#
# The IBEIS dependency cache infrastructure designates three ROOT object types
# for this type of computation: Images, Annotations, and Parts.  There exists
# three parallel decorators that allow one to make a new function for the
# appropriate dependency graph tree.  Adding a new node to the graph requires
# writing a function with the correct inputs and a specific configured decorator.
# We provide a working example below on how to write the decorator for Hash and
# HashSum.
#
# We provide an example below on how to create your own custom depc and
# decorator function
register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']
# register_preproc_part  = controller_inject.register_preprocs['part']


# IBEIS Code Structure, Documentation, Tests, Linting, and Profiling
# ------------------------------------------------------------------
# The IBEIS code based is actually a collection of 17 high-level repositories:
#
#     Core IBEIS repositories
#         ibeis
#         ibeis_cnn  (deprecated soon)
#
#     Core Utility repositories (in order of most use)
#         utool
#         ubelt
#         dtool
#         vtool
#         plottool
#         guitool
#         detecttools
#
#     Core Detection repositories (in order of most use)
#         lightnet
#         brambox
#         pydarknet
#         pyrf
#
#     Core Identification repositories
#         flann
#         hesaff
#
#     First-party Plug-in repositories
#         ibeis-curvrank-module
#         ibeis-flukematch-module
#
# It is important to have all of these repositories installed and configured with
# Python's package manager.  If otherwise specified, the development branch for
# all of these repositories is "next".
#
# Code Documentation
# ------------------
# The IBEIS documentation style uses a modified version of Sphinx "doctests" for
# all documentation and testing.  The ability to write good documentation directly
# in the header of the function is of high value by any contributor to the IBEIS
# code base and any plug-in maintainer.  We provide examples of how to correctly
# document your code throughout this _plugin.py file.
#
# To build the documentation, run the script "build_documentation.sh" in the top-
# level folder of this repository.  The built documentation will be rendered into
# a web HTML format in the folder _page/.  The documentation can be searched and
# navigated by loading _page/index.html.
#
# Code Tests
# ----------
# The immediate benefit of this structure  is that documentation can live alongside
# parameter specifications, REST API endpoints, and unit testing code block(s).
# To run a given test code block, one must simply tell python to execute
# the module and pass in the function name and the test index.  For example, if
# you want to run the first code test for the function
# ibs.ibeis_plugin_identification_example_hello_world() in this file, you can call
#
#     python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_hello_world:0
#
# Note the ":0" index specifier at the end of this Command Line call.  To run all
# of the tests for a specified function, you must remove any post-fix.  For example,
#
#     python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_hello_world
#
# will run all of the tests for that function.  To run all tests for an entire file,
# you can simply call:
#
#     python -m ibeis_plugin_identification_example._plugin --allexamples
#
# We also provide a handy script at the top level path for this repository called
# "run_tests.py" that will execute all of the tests for all files.  A summary
# report "timeings.txt" and "failed_doctests.txt" will be created for each run.
#
# When run in bulk, you may with to disable a specific test from the set of all
# tests you end up writing.  You may do this to improve the speed of the batch
# test call or to exclude tests that are currently a work-in-progress or are
# intended for storing notes and helpful code snippets.  You can add the first-line
# macro "ENABLE_DOCTEST" or "DISABLE_DOCTEST" to enable or disable the test.
#
# Python Code Linting (Flake8)
# ----------------------------
# All IBEIS Python code requires a Linter to be run on any code contributions
# for the main code base.  While we do not explicitly require this for IBEIS
# plug-ins, we STRONGLY suggest using one for all Python code.  We do acknowledge,
# however, that the full Python Flake8 (http://flake8.pycqa.org/en/latest/)
# specification is quire restrictive.  We have a set of allowed errors and warnings
# to ignore, below:
#
#     "D100", "D101", "D102", "D103", "D105", "D200", "D205", "D210", "D400",
#     "D401", "D403", "E127", "E201", "E202", "E203", "E221", "E222", "E241",
#     "E265", "E271", "E272", "E301", "E501", "N802", "N803", "N805", "N806",
#
# Writing good Python code is subjective but a linter will help to make all
# Python contributors follow a consistent set of cleanliness standards.
#
# Python Code Profiling (ut.inject2)
# ----------------------------------
# On top of a large suite of tools, utool (referred to as "ut") also offers
# profiling functions for code segments.  To see this in action, run any Python
# CLI command (e.g. a test) with the extra CLI parameter "--profile" at the end.
# Any function that has the @profile decorator on the function will be profiled
# for run-time efficiency.  For example, running from the CLI:
#
#     python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_file_download:1 --profile
#
# Will run the test of downloading an image from a remote server, check a local
# copy, delete it, then re-download the image.  The output of this call will look
# something like this:
#
#     **[TEST.FINISH] ibeis_plugin_identification_example_file_download -- SUCCESS
#     [util_io] * Writing to text file: timeings.txt
#     L_____________________________________________________________
#     +-------
#     | finished testing fpath='ibeis_plugin_identification_example/ibeis_plugin_identification_example/_plugin.py'
#     | passed 1 / 1
#     L-------
#     Dumping Profile Information
#      -1.00 seconds - None                                             :None:None
#       0.14 seconds - ibeis_plugin_identification_example_file_download:ibeis_plugin_identification_example/ibeis_plugin_identification_example/_plugin.py:536
#     [util_io] * Writing to text file: profile_output.txt
#     [util_io] * Writing to text file: profile_output.<timestamp>.txt
#
# This output is indicating that the function tool 0.14 seconds to finish.
# Inspecting the file "profile_output.txt" shows:
#
#     Timer unit: 1e-06 s
#
#     Line #      Hits         Time  Per Hit   % Time  Line Contents
#     ==============================================================
#     ...
#     586         2         48.0     24.0      0.0      with ut.Timer() as timer:
#     587         2     136968.0  68484.0     99.9          file_filepath = ut.grab_file_url(file_url, appname='ibeis_plugin_identification_example', check_hash=True)
#     588
#     589                                               # ut.Timer() is a handy context that allows for you to quickly get the run-time
#     590                                               # of the code block under its indentation.
#     591         2         18.0      9.0      0.0      print('Download / verification tool %0.02f seconds' % (timer.ellapsed, ))
#     592
#     593                                               # Ensure that the image exists locally
#     594         2         12.0      6.0      0.0      print('File located at: %r' % (file_filepath, ))
#     595         2         38.0     19.0      0.0      assert os.path.exists(file_filepath)
#     596
#     597                                               # Return the download local file's absolute path
#
# This output tells us that we spent 0.1369 seconds and 99.9% of our total run-time
# in this function downloading the file on line 587.  This output is very helpful
# for any developer wishing to optimize the run-time performance of their code.


# Other miscellaneous notes
# -------------------------
#
# Note 1: The IBEIS code base has a constants file for a lot of convenient
# conversions and names of constructs.  This constants module also keeps track of
# very convenient environment variables:
#
#     ibeis.constants.CONTAINERIZED  (Set to True if running inside a Docker container)
#     ibeis.constants.PRODUCTION     (Set to True if running in production mode)
#
# When PRODUCTION is set to True, please observe a restraint in resource utilization
# for system memory, number of concurrent threads, and GPU memory.
#
# Note 2: We suggest to use interactive embedding with utool.embed() whenever
# and whenever possible.  The use of ut.embed() (we commonly import "utool" with
# the shorthand namespace of "ut") is used throughout the IBEIS code base and is
# supremely helpful when debugging troublesome code.  We have set an example below
# that uses ut.embed() in the ibs.ibeis_plugin_identification_example_hello_world()
# documentation.  We highly recommend calling this function's example test and
# play around with the ibs controller object.  The ibs controller supports tab
# completion for all method functions.  For example, when in the embedded iPython
# terminal, you can input:
#
#     In [1]: ibs
#     Out[1]: <IBEISController(testdb_identification) with UUID 1654bdc9-4a14-43f7-9a6a-5f10f2eaa279>
#
#     In [2]: gid_list = ibs.get_valid_gids()
#
#     In [3]: len(gid_list)
#     Out[3]: 69
#
#     In [4]: ibs.get_image_p <tab complete>
#            [  get_image_orientation()            get_image_paths()                  get_image_species_uuids()           ]
#            [  get_image_orientation_str()        get_image_reviewed()               get_image_thumbnail()               ]
#            [< get_image_party_rowids()           get_image_sizes()                  get_image_thumbpath()              >]
#            [  get_image_party_tag()              get_image_species_rowids()         get_image_thumbtup()                ]
#
# This embedded terminal shows all of the IBEIS functions that start with the prefix
# "ibs.get_image_p", for example "ibs.get_image_paths()".  If you are unsure about
# the API specification for this function, you can ask help from Python directly
# in the embedded session.
#
#     In  [5]: help(ibs.get_image_paths)
#     Out [5]: Help on method get_image_paths in module ibeis.control.manual_image_funcs:
#
#              get_image_paths(gid_list) method of ibeis.control.IBEISControl.IBEISController instance
#                  Args:
#                      ibs (IBEISController):  ibeis controller object
#                      gid_list (list): a list of image absolute paths to img_dir
#
#                  Returns:
#                      list: gpath_list
#
#                  CommandLine:
#                      python -m ibeis.control.manual_image_funcs --test-get_image_paths
#
#                  RESTful:
#                      Method: GET
#                      URL:    /api/image/file/path/
#
#                  Example:
#                      >>> # ENABLE_DOCTEST
#                      >>> from ibeis.control.manual_image_funcs import *  # NOQA
#                      >>> import ibeis
#                      >>> # build test data
#                      >>> ibs = ibeis.opendb('testdb1')
#                      >>> #gid_list = ibs.get_valid_gids()
#                      >>> #gpath_list = get_image_paths(ibs, gid_list)
#                      >>> new_gpath = ut.unixpath(ut.grab_test_imgpath('carl.jpg'))
#                      >>> gid_list = ibs.add_images([new_gpath], auto_localize=False)
#                      >>> new_gpath_list = get_image_paths(ibs, gid_list)
#                      >>> ut.assert_eq(new_gpath, new_gpath_list[0])
#                      >>> result = str(new_gpath_list)
#                      >>> ibs.delete_images(gid_list)
#                      >>> print(result)
#
# The embedded session will dump out the doctest (hopefully with parameter
# and example usage documentation) for that controller function.
#
#
#                                 Good luck!
#      Support for IBEIS and this plug-in example is maintained by Wild Me
#             Wild Me is a non-profit located in Portland, OR, USA
#
#                       Please refer any questions to:
#           dev@wildme.org or https://github.com/WildbookOrg/ibeis


@register_ibs_method
@register_api('/api/plugin/example/identification/helloworld/', methods=['GET'])
def ibeis_plugin_identification_example_hello_world(ibs):
    r"""
    A "Hello world!" example for the IBEIS identification plug-in.

    Args:
        ibs (IBEISController):  ibeis controller object
        imgsetid (None):
        require_unixtime (bool):
        reviewed (None):

    Returns:
        list: gid_list

    CommandLine:
        python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_hello_world

        python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_hello_world:0

        python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_hello_world:1

    RESTful:
        Method: GET

        URL:    /api/plugin/example/identification/helloworld/

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> import utool as ut
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> resp = ibs.ibeis_plugin_identification_example_hello_world()
        >>>
        >>> # Result is a special variable in our doctests.  If the last line
        >>> # contains a "result" assignment, then the test checks if the lines
        >>> # specified just below the test are equal to the value of result.
        >>> result = resp + '\n' + ut.repr3({
        >>>     'database'    : ibs.get_db_init_uuid(),
        >>>     'imagesets'   : len(ibs.get_valid_imgsetids()),
        >>>     'images'      : len(ibs.get_valid_gids()),
        >>>     'annotations' : len(ibs.get_valid_aids()),
        >>>     'names'       : len(ibs.get_valid_nids()),
        >>> })
        [ibeis_plugin_identification_example] hello world with IBEIS controller <IBEISController(testdb_identification) with UUID 1654bdc9-4a14-43f7-9a6a-5f10f2eaa279>
        {
            'annotations': 70,
            'database': UUID('1654bdc9-4a14-43f7-9a6a-5f10f2eaa279'),
            'images': 69,
            'imagesets': 7,
            'names': 21,
        }

    Example1:
        >>> # DISABLE_DOCTEST
        >>> import ibeis
        >>> import utool as ut
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> ut.embed()
    """
    args = (ibs, )
    resp = '[ibeis_plugin_identification_example] hello world with IBEIS controller %r' % args
    return resp


@profile
@register_ibs_method
def ibeis_plugin_identification_example_file_download(file_url):
    r"""
    An example of how to download and cache a file on disk from a web-server.
    This function downloads the image to a local application folder and returns
    the local absolute path.

    CommandLine:
        python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_file_download

        python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_file_download:0

        python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_file_download:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_plugin_identification_example._plugin import *  # NOQA
        >>> import utool as ut
        >>> file_url = 'https://cthulhu.dyn.wildme.io/public/data/lena.png'
        >>> file_filepath = ibeis_plugin_identification_example_file_download(file_url)
        >>> file_bytes = open(file_filepath, 'rb').read()
        >>> file_hash_content = ut.hash_data(file_bytes)
        >>> result = file_hash_content
        pgheflebtrskuncufztrynlzpkmkibwg

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_plugin_identification_example._plugin import *  # NOQA
        >>> import utool as ut
        >>> file_url = 'https://cthulhu.dyn.wildme.io/public/data/lena.png'
        >>> file_filepath = ibeis_plugin_identification_example_file_download(file_url)
        >>> # Force a deletion event on this file to force a re-download
        >>> ut.delete(file_filepath)
        >>> file_filepath_ = ibeis_plugin_identification_example_file_download(file_url)
        >>> assert file_filepath == file_filepath_
    """
    # Download the file to the local computer's application cache directory:
    #     For Windows
    #         ~/AppData/Local
    #     For macOS
    #         ~/Library/Caches
    #     For Linux
    #         ~/.cache
    # where the tilde (~) indicates the current logged-in user's home folder.
    # This call will also check for the URL with a postfix of ".md5" and re-download
    # the file if the local file hash (computed on demand) differs from the server's
    # copy.  If no .md5 file exists on the server, this check is skipped.  For example,
    # if the user asks for "https://domain.com/file.txt", then the hash check will
    # ask the server for the value of "https://domain.com/file.txt.md5".
    with ut.Timer() as timer:
        file_filepath = ut.grab_file_url(file_url, appname='ibeis_plugin_identification_example', check_hash=True)

    # ut.Timer() is a handy context that allows for you to quickly get the run-time
    # of the code block under its indentation.
    print('Download / verification tool %0.02f seconds' % (timer.ellapsed, ))

    # Ensure that the image exists locally
    print('File located at: %r' % (file_filepath, ))
    assert os.path.exists(file_filepath)

    # Return the download local file's absolute path
    return file_filepath


class IdentificationExampleImageHashConfig(dt.Config):  # NOQA
    """
    CommandLine:
        python -m ibeis_plugin_identification_example._plugin --test-IdentificationExampleImageHashConfig

        python -m ibeis_plugin_identification_example._plugin --test-IdentificationExampleImageHashConfig:0

        python -m ibeis_plugin_identification_example._plugin --test-IdentificationExampleImageHashConfig:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_plugin_identification_example._plugin import *  # NOQA
        >>> config = IdentificationExampleImageHashConfig()
        >>> result = config.get_cfgstr()
        >>> print(result)
        IdentificationExampleImageHash(hash_algorithm=sha1,hash_rounds=1000000)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_plugin_identification_example._plugin import *  # NOQA
        >>> config = IdentificationExampleImageHashConfig(hash_algorithm='sha256', hash_rounds=100, hash_salt=b'test')
        >>> result = config.get_cfgstr()
        >>> print(result)
        IdentificationExampleImageHash(hash_algorithm=sha256,hash_rounds=100,hash_salt=b'test')
    """
    _param_info_list = [
        ut.ParamInfo('hash_algorithm',  default='sha1',   valid_values=['sha1', 'sha256']),
        ut.ParamInfo('hash_rounds',     default=int(1e6), type_=int),
        ut.ParamInfo('hash_salt',       default=None,     hideif=None),
    ]


@register_preproc_image(
    tablename='IdentificationExampleImageHash', parents=[IMAGE_TABLE],
    colnames=['hash', 'salt'], coltypes=[str, str],
    configclass=IdentificationExampleImageHashConfig,
    fname='identification_example',
    chunksize=4)
def ibeis_plugin_identification_example_image_hash(depc, gid_list, config):
    r"""
    A toy example of creating a crypto-graphically secure (salted) hash of an on-disk image.

    The SQLite3 database location for this dependency cache table is located in
    the loaded IBEIS database folder.  A new table is made automatically named
    "IdentificationExampleImageHash" in the file
    "_ibsdb/_ibeis_cacahe/identification_example.sqlite", as defined by the
    "tablename" and fname" parameters in the above decorator's configuration.

    The colnames and coltypes parameters specify the desired columns that are to be
    created by the dependency cache.  In practice, we serialize all of these results
    and support customized read/write serialization for external assets.  To use a
    custom external reader or writer, specify the coltype for the appropriate
    column as ('extern', load_func, save_func).  For example, if you want to
    store a Numpy array in its full format, you can specify a coltype of
    ('extern', np.load, np.save) or an image with ('extern', cv2.imread, cv2.imwrite)
    or a cPickle through some fancy Utool magic with
    ('extern', ut.partial(ut.load_cPkl, verbose=False), ut.partial(ut.save_cPkl, verbose=False)).
    The depc controller will automatically create a new file on disk and delete
    any external assets when a cached result is invalidated.  The reader function
    must take in an absolute filepath and return the data and the writer must
    take in the data to save and the absolute filepath destination.

    The configclass parameter must be a dt.Config object, as shown above.  The
    _param_info_list attribute of this class defines a list of parameters for this
    depc node.  Thesse parameter names must be unique across the entire dependency
    graph.  Only the relevant parameters in an entire pipeline configuration will
    be passed to this function in the config dictionary variable.

    The chunksize parameter specifies the maximum number of Image IDs (gids)
    that are passed into gid_list.  If a list of gids larger than the chunksize
    is given to the dependency cache, the depc controller will automatically
    batch them into chunks of size chinksize.  You will not get an empty list.

    Reference:
        https://docs.python.org/2/library/hashlib.html#key-derivation

    CommandLine:
        python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_image_hash

    Example:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> import utool as ut
        >>> import numpy as np
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>>
        >>> gid_list = ibs.get_valid_gids()
        >>> gid_list = gid_list[:10]
        >>>
        >>> # Compute (if not already cached) and return the entire rows for this image
        >>> # By default, this command will create a random salt for each image
        >>> value_list = ibs.depc_image.get('IdentificationExampleImageHash', gid_list, None)
        >>> # Return only the 'hash' column (should have already been computed)
        >>> hash_list = ibs.depc_image.get('IdentificationExampleImageHash', gid_list, 'hash')
        >>> print(hash_list)
        >>>
        >>> # Re-compute the hashes but with a specified, deterministic salt
        >>> salt = b'specialsalt'
        >>> config = {'hash_salt': salt}
        >>> hash_list_ = ibs.depc_image.get('IdentificationExampleImageHash', gid_list, 'hash', config=config)
        >>> salt_list = ibs.depc_image.get('IdentificationExampleImageHash', gid_list, 'salt', config=config)
        >>>
        >>> assert np.all([hash != hash_ for hash, hash_ in zip(hash_list, hash_list_)])
        >>> assert np.all([salt_ == salt for salt_ in salt_list])
        >>>
        >>> # Delete results for the specified config and recompute
        >>> ibs.depc_image.delete_property('IdentificationExampleImageHash', gid_list, config=config)
        >>> hash_list_recompute = ibs.depc_image.get('IdentificationExampleImageHash', gid_list, 'hash', config=config)
        >>> assert hash_list_ == hash_list_recompute
        >>>
        >>> # Delete all cached results, regardless of config
        >>> ibs.depc_image.delete_property_all('IdentificationExampleImageHash', gid_list)
        >>>
        >>> result = hash_list_recompute[0]
        b'3006e4db0ed513a0bdb8eda85ee14d5d16ca7165'
    """
    import hashlib
    import binascii

    # Get the IBEIS controller for this database, useful for getting access to other
    # controller functions made for this plug-in but also the built-in adders, getters,
    # setters, and deleters on controller.
    ibs = depc.controller

    # Parameter validation should be done with the Config class for this
    # function, no need to check here.  Simply unpack and rename as needed
    algorithm = config['hash_algorithm']
    rounds    = config['hash_rounds']
    salt      = config['hash_salt']
    rounds = int(rounds)

    # Load the images and compute the PBKDF2 (Password-Based Key Derivation Function 2)
    # hash using HMAC as the pseudorandom function.
    images = ibs.get_images(gid_list)
    for image in tqdm.tqdm(images):
        # Convert the image's pixel content to binary data
        data = image.data.tobytes()
        # Compute a new salt or use a global salt
        salt_ = os.urandom(128) if salt is None else salt
        # Compute the key with the number of rounds (adds time which increases security)
        derived_key = hashlib.pbkdf2_hmac(algorithm, data, salt_, rounds)
        # Convert key to hex data
        hash_ = binascii.hexlify(derived_key)
        # Return the 2-tuple of the same size
        yield (hash_, salt_, )


class IdentificationExampleImageHashSumConfig(dt.Config):  # NOQA
    """
    CommandLine:
        python -m ibeis_plugin_identification_example._plugin --test-IdentificationExampleImageHashSumConfig

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_plugin_identification_example._plugin import *  # NOQA
        >>> config = IdentificationExampleImageHashSumConfig()
        >>> result = config.get_cfgstr()
        >>> print(result)
        IdentificationExampleImageHashSum()
    """
    _param_info_list = [
        ut.ParamInfo('hash_sum_mod', default=None, hideif=None),
    ]


@register_preproc_image(
    tablename='IdentificationExampleImageHashSum', parents=['IdentificationExampleImageHash'],
    colnames=['sum'], coltypes=[int],
    configclass=IdentificationExampleImageHashSumConfig,
    fname='identification_example',
    chunksize=100)
def ibeis_plugin_identification_example_image_hash_sum(depc, image_hash_rowid_list, config):
    r"""
    A toy example of creating a sum for a crypto-graphically secure (salted) hash,
    which is computed by a previous depc node.  The sum of a hash is computed as
    the sum of the ASCII ordinal index for each character of the hash.  The sum
    is maintained for each character in the sequence as a running total, which has
    a modulus operation applied after adding each character.

    The batch size for this function can be much larger because we aren't
    loading image assets (which can be quite memory intensive) into memory.

    CommandLine:
        python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_image_hash_sum

    Example:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> import utool as ut
        >>> import numpy as np
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>>
        >>> gid_list = ibs.get_valid_gids()
        >>> gid_list = gid_list[:10]
        >>>
        >>> # Delete all cached results from the parent table to force the recompute
        >>> # of the full dependency path
        >>> ibs.depc_image.delete_property_all('IdentificationExampleImageHash', gid_list)
        >>>
        >>> # Compute the hash sum, passing in (!important) Image RowIDs
        >>> config = {'hash_salt': b'deterministic', 'hash_sum_mod': 100}
        >>> hash_sum_list = ibs.depc_image.get('IdentificationExampleImageHashSum', gid_list, 'sum', config=config)
        >>>
        >>> # Add a test to convince ourselves the code and test is working correctly
        >>> # Reuse the same gid_list and config, becuase we aren't narcissists
        >>> hash_list = ibs.depc_image.get('IdentificationExampleImageHash', gid_list, 'hash', config=config)
        >>> hash_ = hash_list[0]
        >>> hash_value_list = list(map(int, hash_))
        >>> # Take advantage that our mod is a multiple of our base
        >>> hash_sum = sum(hash_value_list) % 100
        >>> assert hash_sum == hash_sum_list[0]
        >>>
        >>> # Delete all cached results, regardless of config
        >>> ibs.depc_image.delete_property_all('IdentificationExampleImageHashSum', gid_list)
        >>>
        >>> result = hash_sum_list
        [53, 39, 97, 7, 71, 26, 75, 89, 86, 90]
    """
    # Get the configuration
    modulus = config['hash_sum_mod']

    # Instead of loading results from the IBEIS controller using the Image RowIDs,
    # we are instead passed the rowids for the parent of this depc node, which is
    # the 'IdentificationExampleImageHash' depc node we made above.  Therefore,
    # while we always pass a gid_list into depc.get() we will always receive the
    # parent rowids in this function.  We want to ask our depc for the values
    # for the correct table and using the native rowids, thus we use depc.get_native().
    hash_list = depc.get_native('IdentificationExampleImageHash', image_hash_rowid_list, 'hash')

    for hash_ in hash_list:
        # Keep a running total
        total = 0
        for character in hash_:
            # Convert each character in this hash to its Unicode/ASCII integer
            character = int(character)
            # Add it to the total
            total += character
            # If the modulus is None, skip.  Otherwise, mod after each sum
            if modulus is not None:
                total %= modulus

        yield (total, )



class IdentificationExampleImageHashProdConfig(dt.Config):  # NOQA
    """
    CommandLine:
        python -m ibeis_plugin_identification_example._plugin --test-IdentificationExampleImageHashProdConfig

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_plugin_identification_example._plugin import *  # NOQA
        >>> config = IdentificationExampleImageHashProdConfig()
        >>> result = config.get_cfgstr()
        >>> print(result)
        IdentificationExampleImageHashProd(hash_prod_mod=1000)
    """
    _param_info_list = [
        ut.ParamInfo('hash_prod_mod', default=1000),
    ]


@register_preproc_image(
    tablename='IdentificationExampleImageHashProd', parents=['IdentificationExampleImageHash'],
    colnames=['product'], coltypes=[int],
    configclass=IdentificationExampleImageHashProdConfig,
    fname='identification_example',
    chunksize=100)
def ibeis_plugin_identification_example_image_hash_prod(depc, image_hash_rowid_list, config):
    r"""
    A toy example of creating a product for a crypto-graphically secure (salted) hash,
    which is computed by a previous depc node.  It's algorithmic solution is unchanged
    compared to 'IdentificationExampleImageHashSum' except it replaces the operation
    with multiplication over sum.

    In a production plug-in, the operation would ideally be parameterized to
    eliminate code duplication.  This is a quick example to show the dependency
    of a common parent in depc.

    CommandLine:
        python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_image_hash_prod

    Example:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> import utool as ut
        >>> import numpy as np
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>>
        >>> gid_list = ibs.get_valid_gids()
        >>> gid_list = gid_list[:10]
        >>>
        >>> # Compute the hash product
        >>> config = {'hash_salt': b'deterministic', 'hash_prod_mod': 1000}
        >>> hash_prod_list = ibs.depc_image.get('IdentificationExampleImageHashProd', gid_list, 'product', config=config)
        >>>
        >>> # Delete all cached results, regardless of config
        >>> ibs.depc_image.delete_property_all('IdentificationExampleImageHashProd', gid_list)
        >>>
        >>> result = hash_prod_list
        [101, 657, 285, 945, 874, 1, 545, 657, 402, 889]
    """
    # Get the configuration
    modulus = config['hash_prod_mod']
    assert -1e7 <= modulus and modulus <= 1e7, 'modulus should be relatively small (within a million of zero)'

    hash_list = depc.get_native('IdentificationExampleImageHash', image_hash_rowid_list, 'hash')

    for hash_ in hash_list:
        # Keep a running total
        total = 0
        for character in hash_:
            # Convert each character in this hash to its Unicode/ASCII integer
            character = int(character)

            # Let's cheat slightly for multiplication by ensuring strictly positive integers (no zeros allowed in this house)
            character = abs(character)
            character += 1

            # Multiply it to the total
            total *= character
            # Mod after each multiplication, required for this depc node.
            total %= modulus

            # Get out of here, zeros... you're doing this to yourselves
            total += 1

        yield (total, )


class IdentificationExampleOracleRequest(dt.base.VsOneSimilarityRequest):  # NOQA
    """
    This class is the main vehicle for interacting with the IBEIS identification
    pipeline.  It is amazing, mysterious and complex, so let's ignore it.

    ...for now.

    CommandLine:
        python -m ibeis_plugin_identification_example._plugin --test-IdentificationExampleOracleRequest

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_plugin_identification_example._plugin import *  # NOQA
        >>> import ibeis
        >>> import itertools as it
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>>
        >>> aid_list = ibs.get_valid_aids()
        >>> aid_list = aid_list[:10]
        >>> request = IdentificationExampleOracleRequest.new(ibs.depc_annot, aid_list, aid_list)
        >>>
        >>> result = request
        <IdentificationExampleOracleRequest((testdb_identification) nQ=10, nD=10, nP=90 zhafuthnemeinvyi)>
    """

    # Specify the depc table name that this ID Request should query for results
    _tablename = 'IdentificationExampleOracle'

    # If the score for two annotations (annot1, annot2) is symmetric when the score
    # to indicate similarity is the same if annot1 is the query and annot2 is in
    # the reference database compared to when the opposite is the case.  In general,
    # this is rarely the case for approximated methods, but will be true for
    # distance-based calculations like triplet-loss networks.
    _symmetric = True

    @ut.accepts_scalar_input
    def get_fmatch_overlayed_chip(request, aid_list, overlay=True, config=None):
        depc = request.depc
        ibs = depc.controller

        chips = ibs.get_annot_chips(aid_list)

        if overlay:
            # parameter to show the chips without any ID overlay
            pass

        return chips

    def render_single_result(request, cm, aid, **kwargs):
        # We want to allow the algorithm to show a matching result side-by-side
        # to visualize in web and in other API functions.

        # HACK FOR WEB VIEWER
        overlay = kwargs.get('draw_fmatches')
        # Compute the chips for each aid of the features (if any)
        chips = request.get_fmatch_overlayed_chip([cm.qaid, aid], overlay=overlay,
                                                  config=request.config)
        # Stack the images into a single image canvas
        out_img = vt.stack_image_list(chips)

        # Return an np.ndarray in the OpenCV format
        return out_img

    def _get_match_results(request, depc, qaid_list, daid_list, score_list, config):
        r""" converts table results into format for ipython notebook """
        #qaid_list, daid_list = request.get_parent_rowids()
        #score_list = request.score_list
        #config = request.config

        unique_qaids, groupxs = ut.group_indices(qaid_list)
        #grouped_qaids_list = ut.apply_grouping(qaid_list, groupxs)
        grouped_daids = ut.apply_grouping(daid_list, groupxs)
        grouped_scores = ut.apply_grouping(score_list, groupxs)

        ibs = depc.controller
        unique_qnids = ibs.get_annot_nids(unique_qaids)

        # scores
        _iter = zip(unique_qaids, unique_qnids, grouped_daids, grouped_scores)
        for qaid, qnid, daids, scores in _iter:
            dnids = ibs.get_annot_nids(daids)

            # Remove distance to self
            annot_scores = np.array(scores)
            daid_list_ = np.array(daids)
            dnid_list_ = np.array(dnids)

            is_valid = (daid_list_ != qaid)
            daid_list_ = daid_list_.compress(is_valid)
            dnid_list_ = dnid_list_.compress(is_valid)
            annot_scores = annot_scores.compress(is_valid)

            # Hacked in version of creating an annot match object
            match_result = ibeis.AnnotMatch()
            match_result.qaid = qaid
            match_result.qnid = qnid
            match_result.daid_list = daid_list_
            match_result.dnid_list = dnid_list_
            match_result._update_daid_index()
            match_result._update_unique_nid_index()

            grouped_annot_scores = vt.apply_grouping(annot_scores, match_result.name_groupxs)
            name_scores = np.array([np.sum(dists) for dists in grouped_annot_scores])
            match_result.set_cannonical_name_score(annot_scores, name_scores)
            yield match_result

    def postprocess_execute(request, parent_rowids, result_list):
        # Run on the results returned by the depc node function
        # Get the input rowids
        qaid_list, daid_list = list(zip(*parent_rowids))
        # retrieve the matching score results
        score_list = ut.take_column(result_list, 0)
        # Repackage the results by re-balancing the scores as nessecary
        cm_iter = request._get_match_results(request.depc, qaid_list, daid_list,
                                             score_list, request.config)
        # Resolve the iterator here, computing the ChipMatch list on demand
        cm_list = list(cm_iter)
        return cm_list

    def execute(request, *args, **kwargs):
        # Run before calling the depc node function, used to setup the matching process
        kwargs['use_cache'] = False

        # Compute the ID matching results using this algorithm
        result_list = super(IdentificationExampleOracleRequest, request).execute(*args, **kwargs)

        # Check if the query aids (qaids) has been specified
        # If so, only return those results and filter the output
        qaids = kwargs.pop('qaids', None)
        if qaids is not None:
            result_list = [
                result for result in result_list
                if result.qaid in qaids
            ]
        return result_list


class IdentificationExampleOracleConfig(dt.Config):  # NOQA
    """
    CommandLine:
        python -m ibeis_plugin_identification_example._plugin --test-IdentificationExampleOracleConfig

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_plugin_identification_example._plugin import *  # NOQA
        >>> config = IdentificationExampleOracleConfig()
        >>> result = config.get_cfgstr()
        >>> print(result)
        IdentificationExampleOracle(oracle_fallibility=0.1)
    """
    def get_param_info_list(self):
        return [
            ut.ParamInfo('oracle_fallibility', 0.1),
        ]


@register_preproc_annot(
    tablename='IdentificationExampleOracle', parents=[ANNOTATION_TABLE, ANNOTATION_TABLE],
    colnames=['score'], coltypes=[float],
    configclass=IdentificationExampleOracleConfig,
    requestclass=IdentificationExampleOracleRequest,
    fname='identification_example',
    chunksize=None)
def ibeis_plugin_identification_example_oracle(depc, qaid_list, daid_list, config):
    r"""
    This function is called automatically by the IdentificationExampleOracleRequest
    whenever the appropriate ID algorithm 'IdentificationExampleOracle' is specified
    as the 'pipeline_root'.  No matter the respective size of the original qaid_list
    (Query Annotation RowIDs) and daid_list (Database Annotation RowIDs), this
    function will always be given parallel lists.  The lists are defined as the
    total number of combinations between the query and database annotations, which
    obviously may be very large for big ID queries.

    We thus highly suggest using batch processing and as many database loads as
    possible.  To achieve the original qaid_list and daid_list, simply do the
    following:

    ```
        original_qaid_list = list(set(qaid_list))
        original_daid_list = list(set(daid_list))
    ```

    However, the results are required to be passed in for the appropriate order
    as defined by the qaid_list and daid_list passed to this function.  If your
    call would benefit greatly from creating a single global cache or model for
    this ID call, we strongly suggest making the chunksize=None so that your
    plug-in code can know about the entire ID graph.  Any functions within here
    could rely on batched processing, but there is a benefit to the plug-in knowing
    the context and scale of the ID job it has been asked to complete.

    Our example here will always produce a ID match score of 1.0 for correct matches
    (based on the ground-truth name in the database for each annotation) and 0.0
    for any annotation pair that is of different individuals.  There is a
    configurable 'oracle_fallibility' parameter, which makes a randomized
    determination for each pair to flip this correct decision and make an error.

    From web, this function is called by Wildbook as
        - REST POST to /api/engine/query/graph/
            - Configured pipeline_root specifies algorithm
        - Resoles to Python function start_identify_annots_query
        - Calls Python function query_chips_graph (Endpoint for /api/query/graph/)

    CommandLine:
        python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_oracle:0

        python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_oracle:0 --show

        python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_oracle:1

        python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_oracle:2

        python -m ibeis_plugin_identification_example._plugin --test-ibeis_plugin_identification_example_oracle:3

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_plugin_identification_example._plugin import *  # NOQA
        >>> import ibeis
        >>> import itertools as it
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>>
        >>> depc = ibs.depc_annot
        >>> imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(['Grevy\'s Zebra Query', 'Grevy\'s Zebra Database'])
        >>> qaid_list, daid_list = ibs.get_imageset_aids(imageset_rowid_list)
        >>> config = IdentificationExampleOracleConfig()
        >>>
        >>> # Call function via request
        >>> request = IdentificationExampleOracleRequest.new(depc, qaid_list, daid_list)
        >>> am_list = request.execute()
        >>>
        >>> # Plot the ID results
        >>> ut.quit_if_noshow()
        >>> am = am_list[0]
        >>> am.ishow_analysis(request)
        >>> ut.show_if_requested()

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis_plugin_identification_example._plugin import *  # NOQA
        >>> import ibeis
        >>> import itertools as it
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>>
        >>> depc = ibs.depc_annot
        >>> imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(['Grevy\'s Zebra Query', 'Grevy\'s Zebra Database'])
        >>> qaid_list, daid_list = ibs.get_imageset_aids(imageset_rowid_list)
        >>>
        >>> # HotSpotter, default settings
        >>> query_config_dict = {'pipeline_root': 'IdentificationExampleOracle'}
        >>> result_dict = ibs.query_chips_graph(qaid_list, daid_list, query_config_dict=query_config_dict)
        >>> print(result_dict)

    Example2:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_plugin_identification_example._plugin import *  # NOQA
        >>> import ibeis
        >>> import itertools as it
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>>
        >>> depc = ibs.depc_annot
        >>> imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(['Grevy\'s Zebra Query', 'Grevy\'s Zebra Database'])
        >>> qaid_list, daid_list = ibs.get_imageset_aids(imageset_rowid_list)
        >>>
        >>> # HotSpotter, default settings
        >>> query_config_dict = {}
        >>> result_dict = ibs.query_chips_graph(qaid_list, daid_list, query_config_dict=query_config_dict)

    Example3:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_plugin_identification_example._plugin import *  # NOQA
        >>> import ibeis
        >>> import itertools as it
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>>
        >>> depc = ibs.depc_annot
        >>> imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(['Fluke Query', 'Fluke Database'])
        >>> qaid_list, daid_list = ibs.get_imageset_aids(imageset_rowid_list)
        >>>
        >>> # Dynamic Time Warping (BC), default settings
        >>> # query_config_dict = {'pipeline_root' : 'BC_DTW'}
        >>>
        >>> # Dynamic Time Warping (OC), default settings
        >>> # query_config_dict = {'pipeline_root' : 'OC_WDTW'}
        >>>
        >>> # CurvRank (Fluke), default settings
        >>> query_config_dict = {'pipeline_root' : 'CurvRankFluke'}
        >>>
        >>> result_dict = ibs.query_chips_graph(qaid_list, daid_list, query_config_dict=query_config_dict)

    Example4:
        >>> # DISABLE_DOCTEST
        >>> from ibeis_plugin_identification_example._plugin import *  # NOQA
        >>> import ibeis
        >>> import itertools as it
        >>> from ibeis.init import sysres
        >>> dbdir = sysres.ensure_testdb_identification_example()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>>
        >>> depc = ibs.depc_annot
        >>> imageset_rowid_list = ibs.get_imageset_imgsetids_from_text(['Dorsal Query', 'Dorsal Database'])
        >>> qaid_list, daid_list = ibs.get_imageset_aids(imageset_rowid_list)
        >>>
        >>> # CurvRank (Dorsal), default settings
        >>> query_config_dict = {'pipeline_root' : 'CurvRankDorsal'}
        >>> result_dict = ibs.query_chips_graph(qaid_list, daid_list, query_config_dict=query_config_dict)
    """
    import random
    ibs = depc.controller

    # Error rate
    error = config['oracle_fallibility']

    # Retrieve the name for the query and database annotations.
    # This is a great example of making a global cached call to the IBEIS controller
    # that will be re-used multiple times in memory.  It would be a major slowdown
    # to call the get_annot_nids call for every single pair as they are encountered
    # below in the for loop.  Do this in a single batch instead and on the unique
    # set of RowIDs instead of a list with (probably) a lot of duplicates.
    original_qaid_list = list(set(qaid_list))
    original_daid_list = list(set(daid_list))

    original_qnid_list = ibs.get_annot_nids(original_qaid_list)
    original_dnid_list = ibs.get_annot_nids(original_daid_list)

    original_qnid_dict = dict(zip(original_qaid_list, original_qnid_list))
    original_dnid_dict = dict(zip(original_daid_list, original_dnid_list))

    args = (len(original_qaid_list), len(original_daid_list), )
    print('Running ID on %d query annotations against %d database annotations' % args)

    pair_list = list(zip(qaid_list, daid_list))

    for qaid, daid in tqdm.tqdm(pair_list):
        # Retrieve GT name for annotations
        qnid = original_qnid_dict.get(qaid, None)
        dnid = original_dnid_dict.get(daid, None)

        # Check if the names are the same
        result = qnid == dnid

        # Flip the result, randomly
        if random.uniform(0.0, 1.0) <= error:
            result = not result

        # 1.0 for a prediction of same, 0.0 for different
        score = 1.0 if result else 0.0

        yield (score, )


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_plugin_identification_example._plugin --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
