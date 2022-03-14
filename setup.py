from setuptools import setup, find_packages

setup(
    name='CAGMon',
    version='0.8.5',
    author='Phil Jung',
    author_email='pjjung@nims.re.kr',
    description='Correlation Analysis based on Glitch Monitoring',
    url='https://github.com/pjjung/cagmon',
    packages=find_packages(exclude=['test','examples]),
    entry_points={
        "console_scripts": [
            "cagmon = cagmon.main:main"
        ]
    },
    install_requires=["setuptools",
                      "gwpy>=1.0.1",
                      "lalsuite>=1.4.4",
                      "minepy>=1.2.5"],
    python_requires=">=2.7",
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
    ],
)
