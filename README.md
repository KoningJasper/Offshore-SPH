# Offshore-SPH

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/14d8416e36b14a98bd533dcfb2f4166a)](https://app.codacy.com/app/KoningJasper/Offshore-SPH?utm_source=github.com&utm_medium=referral&utm_content=KoningJasper/Offshore-SPH&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://travis-ci.org/KoningJasper/Offshore-SPH.svg?branch=master)](https://travis-ci.org/KoningJasper/Offshore-SPH)
[![codecov](https://codecov.io/gh/KoningJasper/Offshore-SPH/branch/master/graph/badge.svg)](https://codecov.io/gh/KoningJasper/Offshore-SPH)

A study in offshore use of smoothed particle hydrodynamics (SPH). This project aims to provide a simple and interactive SPH simulator, complete with results viewing, and boundary condition generation specifically designed for the offshore industry. It focusses on 2D interactions, however, it can be extended to cover 3D scenarios.

SPH is a simulation method originally developed by J.A. Monaghan in his 1977 paper. It is a langrangian simulation method that does not require the use of a mesh. Instead an gas, fluid or solid is simulated by (macro) particles. These particles discretize the objects properties and provide interpolation points for the simulation.

The two main forms of fluid SPH are implemented in this program: `WCSPH` and `ICSPH`. WCSPH stands for weakly compressible SPH, it assumes the fluid to be slightly compressible and is the original SPH method for the simulation of fluids. It's very easy to implement and captures phenomenon occuring in medium speed well. It's suitable for simulation of dam-breaks and other low speed fluid problems.

ICSPH or Incompressible SPH differs from WCSPH in that it treats the fluid as in-compressible. This in a generally correct assumption since the density of water hardly increases with added pressure in real life.

## Required Software

The following software is required to run the project and get all the results.

- Python 3.6 (64-bit)
- ffmpeg (for the generation of animations)

All python packages are contained in the requirements.txt file. These can be installed using the following command.

```ps
pip install -r requirements.txt
```

## Build and Test

Tests are contained in the `test` folder and follow the same structure as the source (`src`) folder. Most of the code is tested in these unit tests. In the unit tests implementations are verified against known data sources such as papers and other SPH implementations such as PySPH.

## Examples

Examples are contained in the `examples` directory. These examples contain validation cases such as a 2D dam break, boundary condition verification, and other SPH research replication. All the examples create animations as output and therefore require `ffmpeg` to be installed, ffmpeg can be acquired from the following location.

Examples can be run by using the following command in the root folder of the project.
```sh
python -m examples.dam_break_complete
```

Running this example should only take a couple of minutes and create a ``dam_break_simple.mp4`` video in the root folder of the project.
