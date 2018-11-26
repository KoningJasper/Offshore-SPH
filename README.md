# Offshore-SPH

A study in offshore use of smoothed particle hydrodynamics (SPH). This project aims to provide a simple and interactive SPH simulator, complete with results viewing, and boundary condition generation specifically designed for the offshore industry. It focusses on 2D interactions, however, it can be extended to cover 3D scenarios.

SPH is a simulation method originally developed by J.A. Monaghan in his 1977 paper. It is a langrangian simulation method that does not require the use of a mesh. Instead an gas, fluid or solid is simulated by (macro) particles. These particles discretize the objects properties and provide interpolation points for the simulation.

The two main forms of fluid SPH are implemented in this program: `WCSPH` and `ICSPH`. WCSPH stands for weakly compressible SPH and

## Required Software

The following software is required to run the project and get all the results.

- Python 3.7
- ffmpeg (for the generation of animations)

All python packages are contained in the requirements.txt file. These can be installed using the following command.

```ps
pip install -r requirements.txt
```

## Build and Test

Tests are contained in the `test` folder and follow the same structure as the source (`src`) folder. Most of the code is tested in these unit tests. In the unit tests implementations are verified against known data sources such as papers and other SPH implementations such as PySPH.

## Examples

Examples are contained in the `example` directory. These examples contain validation cases such as a 2D dam break, boundary condition verification, and other SPH research replication. All the examples create animations as output and therefore require `ffmpeg` to be installed, ffmpeg can be acquired from the following location.
