FROM ubuntu:24.04 AS builder

# Env variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="/code/SuperBuild/install/lib/python3.12/dist-packages:/code/SuperBuild/install/bin/opensfm" \
    LD_LIBRARY_PATH="/code/SuperBuild/install/lib"

# Prepare directories
WORKDIR /code

# HikeAndMap: first only what the long build needs (the build script, the Python requirements and the SuperBuild), then
# the rest of the code. A change to ODX's own Python code then rebuilds in minutes from the cached SuperBuild; a change to
# requirements.txt or SuperBuild/ (for instance the commit of one of our forks) still rebuilds everything, as it must.
COPY configure.sh requirements.txt ./
COPY SuperBuild ./SuperBuild
COPY docker ./docker

# Run the build
RUN bash configure.sh install

# Clean Superbuild
RUN bash configure.sh clean

# Copy everything else (SuperBuild's build output is in .dockerignore, so it is not overwritten)
COPY . ./

### END Builder

### Use a second image for the final asset to reduce the number and
# size of the layers.
FROM ubuntu:24.04

# Env variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH="/code/SuperBuild/install/lib/python3.12:/code/SuperBuild/install/bin/opensfm" \
    LD_LIBRARY_PATH="/code/SuperBuild/install/lib" \
    PDAL_DRIVER_PATH="/code/SuperBuild/install/bin"

WORKDIR /code

# Copy everything we built from the builder
COPY --from=builder /code /code

# Copy the Python libraries installed via pip from the builder
COPY --from=builder /usr/local /usr/local

# Install shared libraries that we depend on via APT, but *not*
# the -dev packages to save space!
# Also run a smoke test on ODX and OpenSfM
RUN bash configure.sh installruntimedeps \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
  && bash run.sh --help \
  && bash -c "eval $(python3 /code/modules/context.py) && python3 -c 'from opensfm import io, pymap'"

# Entry point
ENTRYPOINT ["python3", "/code/run.py"]
