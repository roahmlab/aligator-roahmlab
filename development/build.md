# Build and develop with pixi

The easiest way to set up a development environment is to use [pixi](https://pixi.sh/latest/#installation).

[pixi](https://pixi.sh/latest/) is a cross-platform package manager for developers.
It installs all required dependencies in the `.pixi` directory.
It's used by our CI, so you get the same stable and tested dependencies.

Run the following command to install dependencies, configure, build and test the project:

```bash
pixi run test
```

The project is built in the `build` directory.

The typical workflow is:

```bash
pixi shell
pixi run configure
ninja -C build
```

After `pixi run configure`, use `cmake` and `ninja` manually to reconfigure and build the project.

## Environments

The pixi manifest contains many environments. The most common ones are:

- **default**: core aligator and Python bindings
- **all**: all aligator features and Python bindings
- **all-with-croco**: all aligator features, crocoddyl support and Python bindings

To activate a specific environment, run:

```bash
pixi shell -e all-with-croco
```

Using **all-with-croco** makes it easy to choose which features to build.
In this case, use the following CMake options:

- `BUILD_WITH_PINOCCHIO_SUPPORT`: Pinocchio support
- `BUILD_CROCODDYL_COMPAT`: Crocoddyl support
- `BUILD_WITH_OPENMP_SUPPORT`: OpenMP support
- `BUILD_BENCHMARKS`: Benchmark
- `BUILD_EXAMPLES`: Examples
- `BUILD_PYTHON_INTERFACE`: Python bindings
- `GENERATE_PYTHON_STUBS`: Python stubs generation

With the **all-with-croco** environment, all these options are ON.
To turn one off, pass the corresponding `-D` flag to `cmake`:

```bash
cmake -B build -DBUILD_PYTHON_INTERFACE=OFF
```
