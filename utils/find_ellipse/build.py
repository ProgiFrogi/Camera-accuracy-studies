import os
import subprocess
import sys

def build():
    if True:
        current_working_directory = os.getcwd()
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        os.chdir(current_directory)
        # os.system("python setup.py build_ext --quiet --inplace")
        python_interpreter_path = sys.executable
        subprocess.check_output(python_interpreter_path+' setup.py build_ext --quiet --inplace', shell=True)
        os.chdir(current_working_directory)
    if not os.path.isfile(os.path.dirname(os.path.abspath(__file__)) + "/find_ellipse_source.c"):
        raise Exception("if you see this exception,then you need to open this file and run it")

build()