FROM ubuntu:22.04
LABEL title="ml_near_well"
LABEL description="Docker image to reproduce results in 'A Machine-Learned Near-Well Model in OPM Flow'"
LABEL version="0.1"
LABEL maintainer="Peter von Schultzendorff"
LABEL email="peter.schultzendorff@uib.no"

# Suppress interactive dialogue during package installation.
ARG DEBIAN_FRONTEND=noninteractive
ARG OPM_REF=release/2025.10
ARG OPM_BUILD_JOBS=5

# Switch to root user to install packages.
USER root

# Add OPM PPA so supported Dune and prerequisite packages are available on Ubuntu 22.04.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    dirmngr \
    gnupg \
    software-properties-common && \
    add-apt-repository -y ppa:opm/ppa && \
    apt-get update

# Install build and runtime dependencies for OPM source builds and Python workflows.
RUN apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cm-super-minimal \
    cmake \
    dvipng \
    gfortran \
    git \
    libcjson-dev \
    libblas-dev \
    libboost-all-dev \
    libdune-common-dev \
    libdune-geometry-dev \
    libdune-grid-dev \
    libdune-istl-dev \
    libfmt-dev \
    liblapack-dev \
    mpi-default-bin \
    mpi-default-dev \
    libsuitesparse-dev \
    libtrilinos-zoltan-dev \
    pkg-config \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-venv \
    python3-pip \
    texlive \
    texlive-fonts-recommended \
    texlive-latex-extra \
    zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Create a non-root user to run the reproducibility workflow. Set up the ML_near_well 
# and pyopmnearwell repositories and install their Python dependencies. 
RUN useradd -ms /bin/bash diligent_researcher
USER diligent_researcher

ENV HOME=/home/diligent_researcher
WORKDIR $HOME

# Copy ML_near_well repository into the image and install Python dependencies.
COPY . ./ML_near_well
RUN cd ML_near_well && \
   python3.10 -m pip install -r requirements.txt

# Clone and install pyopmnearwell.
RUN git clone --branch 2024-08_ML_near_well_article https://github.com/cssr-tools/pyopmnearwell && \
   cd pyopmnearwell && \
   python3.10 -m pip install -e .


# Switch back to root user to clone and build OPM from source. The OPM files are
# slightly modified to include the ML near-well model.

USER root

ENV OPM_ROOT=/opt/opm_src
ENV OPM_PATH=$OPM_ROOT
ENV FLOW_PATH=$OPM_ROOT/opm-simulators/build/bin/flow

# Clone OPM following the documented sibling-repository layout.
WORKDIR $OPM_ROOT

RUN for repo in opm-common opm-grid opm-simulators opm-upscaling; do \
        git clone --branch "$OPM_REF" --depth 1 "https://github.com/OPM/${repo}.git"; \
    done

# Build OPM components in the correct order.
RUN mkdir -p "$OPM_ROOT/opm-common/build" && \
    cd "$OPM_ROOT/opm-common/build" && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$OPM_BUILD_JOBS

RUN mkdir -p "$OPM_ROOT/opm-grid/build" && \
    cd "$OPM_ROOT/opm-grid/build" && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$OPM_BUILD_JOBS

# Copy modified OPM ML near-well model files from ML_near_well repository before
# building the simulators.
COPY runscripts/ECMOR24_proceeding/h2o/FlowProblemParameters.cpp ./opm-simulators/opm/simulators/flow/FlowProblemParameters.cpp
COPY runscripts/ECMOR24_proceeding/h2o/FlowProblemParameters.hpp ./opm-simulators/opm/simulators/flow/FlowProblemParameters.hpp
COPY runscripts/ECMOR24_proceeding/h2o/MLNearWellConfig.hpp ./opm-simulators/opm/simulators/wells/MLNearWellConfig.hpp
COPY runscripts/ECMOR24_proceeding/h2o/StandardWell.hpp ./opm-simulators/opm/simulators/wells/StandardWell.hpp
COPY runscripts/ECMOR24_proceeding/h2o/StandardWell_impl.hpp ./opm-simulators/opm/simulators/wells/StandardWell_impl.hpp

RUN mkdir -p "$OPM_ROOT/opm-simulators/build" && \
    cd "$OPM_ROOT/opm-simulators/build" && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$OPM_BUILD_JOBS flow_gaswater_dissolution_diffuse

RUN mkdir -p "$OPM_ROOT/opm-upscaling/build" && \
    cd "$OPM_ROOT/opm-upscaling/build" && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$OPM_BUILD_JOBS

RUN ln -sf "$OPM_ROOT/opm-simulators/build/bin/flow" /usr/local/bin/flow && \
    ln -sf "$OPM_ROOT/opm-common/build/bin/co2brinepvt" /usr/local/bin/co2brinepvt


# Ensure standard install locations and source-built Flow are on PATH.
RUN echo 'export PATH="/usr/local/bin:/usr/bin:$PATH"' >> "$HOME/.bashrc"

# Switch back to non-root user and run the reproducibility workflow.
USER diligent_researcher
WORKDIR $HOME
CMD ["bash", "-lc", "cd ./ML_near_well && bash runscripts/run.bash"]