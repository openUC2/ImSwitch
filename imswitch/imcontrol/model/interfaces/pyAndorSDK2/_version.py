# major should never change
# minor indicates changes within code - to update run version_updater.py with --minor
# build indicates updates to dlls - to update run version_updater.py with --build

major = 1
minor = 2
build = 0
__version_info__ = (major, minor, build)
__version__ = '.'.join(map(str, __version_info__))