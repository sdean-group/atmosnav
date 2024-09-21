from setuptools import setup, find_packages
setup(
    name='atmosnav',
    version='0.1',
    description='Simulation and learning framework for atmospheric navigation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Myles Pasetsky, Janna Lin, Bradley Guo',
    author_email='mhp58@cornell.edu, jnl77@cornell.edu, bzg4@cornell.edu',
    packages=find_packages(include=['atmosnav*']),
#    python_requires='>=3.12.0',
    install_requires = ['matplotlib','tropycal','shapely','cartopy','jax','jaxlib'],
    extras_require={},
    entry_points={},
)