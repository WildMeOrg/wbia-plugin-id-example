#!/usr/bin/env python
from __future__ import absolute_import, print_function, division
import subprocess
import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


CLASSIFIERS = '''
Development Status :: 3 - Alpha
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: Apache Software License
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Programming Language :: Python :: 3.3
Programming Language :: Python :: 3.4
Topic :: Scientific/Engineering :: Bio-Informatics
'''
NAME                = 'IBEIS Plugin Example'
MAINTAINER          = 'Wildbook Org. | IBEIS IA'
MAINTAINER_EMAIL    = 'info@wildme.org'
DESCRIPTION         = 'An example of how to design and use a Python module ' + \
                      'as a plugin in the IBEIS IA system'
LONG_DESCRIPTION    = DESCRIPTION
KEYWORDS            = ['ibeis', 'plugin', 'wildbook', 'ia']
URL                 = 'https://github.com/WildbookOrg/'
DOWNLOAD_URL        = ''
LICENSE             = 'Apache'
AUTHOR              = MAINTAINER
AUTHOR_EMAIL        = MAINTAINER_EMAIL
PLATFORMS           = ['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix']
MAJOR               = 0
MINOR               = 1
MICRO               = 0
SUFFIX              = 'dev0'
VERSION             = '%d.%d.%d.%s' % (MAJOR, MINOR, MICRO, SUFFIX)
PACKAGES            = ['ibeis_plugin_example']


def git_version():
    """
    Return the sha1 of local git HEAD as a string.
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'PYTHONPATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            env=env
        ).communicate()[0]
        return out
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = 'unknown-git'
    return git_revision


def write_version_py(filename=os.path.join('ibeis_plugin_example', 'version.py')):
    cnt = '''
# THIS FILE IS GENERATED FROM SETUP.PY
version = '%(version)s'
git_revision = '%(git_revision)s'
full_version = '%%(version)s.%%(git_revision)s' %% {
    'version': version,
    'git_revision': git_revision,
}
'''
    FULL_VERSION = VERSION
    if os.path.isdir('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists(filename):
        GIT_REVISION = 'RELEASE'
    else:
        GIT_REVISION = 'unknown'

    FULL_VERSION += '.' + GIT_REVISION
    text = cnt % {
        'version': VERSION,
        'git_revision': GIT_REVISION
    }
    try:
        with open(filename, 'w') as a:
            a.write(text)
    except Exception as e:
        print(e)


def do_setup():
    write_version_py()
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        classifiers=CLASSIFIERS,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        platforms=PLATFORMS,
        packages=PACKAGES,
        keywords=CLASSIFIERS.replace('\n', ' ').strip(),
    )


if __name__ == '__main__':
    do_setup()
