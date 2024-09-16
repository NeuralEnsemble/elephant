import platform
import sys

from setuptools import setup, Extension
from setuptools.command.install import install
from setuptools.command.develop import develop

if platform.system() == "Windows":
    fim_module = Extension(
        name='elephant.spade_src.fim',
        sources=['elephant/spade_src/src/fim.cpp'],
        include_dirs=['elephant/spade_src/include'],
        language='c++',
        libraries=[],
        extra_compile_args=[
            '-DMODULE_NAME=fim', '-DUSE_OPENMP', '-DWITH_SIG_TERM',
            '-Dfim_EXPORTS', '-fopenmp', '/std:c++17'],
        #optional=True
    )
elif platform.system() == "Darwin":
    fim_module = Extension(
        name='elephant.spade_src.fim',
        sources=['elephant/spade_src/src/fim.cpp'],
        include_dirs=['elephant/spade_src/include'],
        language='c++',
        libraries=['pthread', 'omp'],
        extra_compile_args=[
            '-DMODULE_NAME=fim', '-DUSE_OPENMP', '-DWITH_SIG_TERM',
            '-Dfim_EXPORTS', '-O3', '-pedantic', '-Wextra',
            '-Weffc++', '-Wunused-result', '-Werror', '-Werror=return-type',
            '-Xpreprocessor',
            '-fopenmp', '-std=gnu++17'],
        #optional=True
    )
elif platform.system() == "Linux":
    fim_module = Extension(
        name='elephant.spade_src.fim',
        sources=['elephant/spade_src/src/fim.cpp'],
        include_dirs=['elephant/spade_src/include'],
        language='c++',
        libraries=['pthread', 'gomp'],
        extra_compile_args=[
            '-DMODULE_NAME=fim', '-DUSE_OPENMP', '-DWITH_SIG_TERM',
            '-Dfim_EXPORTS', '-O3', '-pedantic', '-Wextra',
            '-Weffc++', '-Wunused-result', '-Werror',
            '-fopenmp', '-std=gnu++17'],
        #optional=True
    )

setup_kwargs = {}

# no compile options and corresponding extensions
options = {"--no-compile": None, "--no-compile-spade": fim_module}
# check if any option was specified
if not any([True for key in options.keys() if key in sys.argv]):
    if platform.system() in ["Windows", "Linux"]:
        setup_kwargs["ext_modules"] = [fim_module]
else:  # ...any option was specified
    # select extensions accordingly
    extensions = [module for flag, module in options.items() if
                  flag not in sys.argv]
    if None in extensions:  # None indicates "--no-compile" not in sys.argv
        extensions.remove(None)
        setup_kwargs["ext_modules"] = extensions


class CommandMixin(object):
    """
    This class acts as a superclass to integrate new commands in setuptools.
    """
    user_options = [
        ('no-compile', None, 'do not compile any C++ extension'),
        ('no-compile-spade', None, 'do not compile spade related C++ extension')  # noqa
    ]

    def initialize_options(self):
        """
        The method is responsible for setting default values for
        all the options that the command supports.

        Option dependencies should not be set here.
        """

        super().initialize_options()
        # Initialize options
        self.no_compile_spade = None
        self.no_compile = None

    def finalize_options(self):
        """
        Overriding a required abstract method.

        This method is responsible for setting and checking the
        final values and option dependencies for all the options
        just before the method run is executed.

        In practice, this is where the values are assigned and verified.
        """

        super().finalize_options()

    def run(self):
        """
        Sets global which can later be used in setup.py to remove c-extensions
        from setup call.
        """
        # Use options
        global no_compile_spade
        global no_compile
        no_compile_spade = self.no_compile_spade
        no_compile = self.no_compile

        super().run()


class InstallCommand(CommandMixin, install):
    """
    This class extends setuptools.command.install class, adding user options.
    """
    user_options = getattr(
        install, 'user_options', []) + CommandMixin.user_options


class DevelopCommand(CommandMixin, develop):
    """
    This class extends setuptools.command.develop class, adding user options.
    """
    user_options = getattr(
        develop, 'user_options', []) + CommandMixin.user_options


# add classes to setup-kwargs to add the user options
setup_kwargs['cmdclass'] = {'install': InstallCommand,
                            'develop': DevelopCommand}

setup(**setup_kwargs)
