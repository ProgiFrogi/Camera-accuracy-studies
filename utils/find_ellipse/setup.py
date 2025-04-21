# from Cython.Compiler.Options import annotate
from setuptools import setup,Extension
from Cython.Build import cythonize
# from setuptools.command.build_ext import build_ext
# from setuptools.config.expand import cmdclass
# import os
# import shutil

ext = Extension(
    name = "find_ellipse",
    sources=["find_ellipse_source.py"]
)

# Custom build_ext command to specify build directory
# class custom_build_ext(build_ext):
#
#     path_to_dir = 'utils/find_ellipse/'
#
#     def finalize_options(self):
#         build_ext.finalize_options(self)
#         # Set the build library directory to your desired path
#         self.build_lib = self.path_to_dir
#
#     def run(self):
#         # Run the standard build process
#         build_ext.run(self)
#
#         # Move the built extension to the desired directory
#         build_dir = self.build_lib
#         output_dir = self.path_to_dir
#         for output in self.outputs:
#             # Get the filename from the output path
#             filename = os.path.basename(output)
#             # Move the file to the desired output directory
#             shutil.move(output, os.path.join(output_dir, filename))

setup(
    ext_modules=cythonize(ext,
                          annotate=True,
                          quiet=True
                          ),
    # cmdclass={'build_ext':custom_build_ext}

)