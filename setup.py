from setuptools import setup

entry_points = {'console_scripts': ['runproc = sourcesim.sourcesim:process_run']}

setup(
    name = "sourcesim",
    version = "0.1",
    packages = ["sourcesim"],
    url = "http://github.com/lewyh/sourcesim",
    license = "MIT",
    author = "Hywel Farnhill",
    author_email = "hywel.farnhill@gmail.com",
    entry_points = entry_points,
    install_requires = ['numpy>=1.9',
                        'astropy>=0.3'],
    package_data = {'sourcesim': ['data/*']}
)