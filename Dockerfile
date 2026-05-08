FROM ubuntu:22.04
LABEL org.opencontainers.image.title="ml_near_well"
LABEL org.opencontainers.image.description="Reproducibility image for ML near-well OPM Flow experiments"
LABEL org.opencontainers.image.version="0.1"
LABEL org.opencontainers.image.authors="Peter von Schultzendorff <peter.schultzendorff@uib.no>"

# Suppress interactive dialogue during package installation.
ARG DEBIAN_FRONTEND=noninteractive
ARG OPM_REF=release/2025.10
ARG OPM_BUILD_JOBS=8
ARG DUNE_VERSION=v2.9.1

# Switch to root user to install packages.
USER root

# Install build and runtime dependencies for OPM source builds and Python workflows.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    gfortran \
    git \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    libboost-all-dev \
    libsuitesparse-dev \
    libtrilinos-zoltan-dev \
    libfmt-dev \
    libcjson-dev \
    libfmt8 \
    libcjson1 \
    mpi-default-bin \
    mpi-default-dev \
    python3.10 \
    python3.10-venv \
    python3-pip \
    texlive \
    texlive-fonts-recommended \
    texlive-latex-extra \
    dvipng \
    cm-super-minimal \
    zlib1g-dev \
    dirmngr \
    gnupg \
    software-properties-common \
 && add-apt-repository -y ppa:opm/ppa \
 && apt-get update \
 && rm -rf /var/lib/apt/lists/*

# Build DUNE from source.
ENV DUNE_ROOT=/opt/dune
ENV DUNE_INSTALL=${DUNE_ROOT}/install

WORKDIR ${DUNE_ROOT}

RUN for mod in dune-common dune-geometry dune-grid dune-istl; do \
      git clone --branch ${DUNE_VERSION} --depth 1 \
        https://gitlab.dune-project.org/core/$mod.git; \
    done

RUN for mod in dune-common dune-geometry dune-grid dune-istl; do \
      cmake -S ${mod} -B ${mod}/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=${DUNE_INSTALL} && \
      cmake --build ${mod}/build -j ${OPM_BUILD_JOBS} && \
      cmake --install ${mod}/build && \
      rm -rf ${mod}/build; \
    done


ENV CMAKE_PREFIX_PATH=${DUNE_ROOT}/install:${CMAKE_PREFIX_PATH}
ENV PKG_CONFIG_PATH=${DUNE_ROOT}/install/lib/pkgconfig:${PKG_CONFIG_PATH}
ENV LD_LIBRARY_PATH=${DUNE_ROOT}/install/lib:${LD_LIBRARY_PATH}


# Build OPM from source.
ENV OPM_ROOT=/opt/opm_src
ENV OPM_BUILD=/opt/opm_build

WORKDIR ${OPM_ROOT}

RUN for repo in opm-common opm-grid opm-simulators opm-upscaling; do \
      git clone --branch ${OPM_REF} --depth 1 \
      https://github.com/OPM/${repo}.git; \
    done

# Modify OPM source files to include ML near-well model.
COPY runscripts/ECMOR24_proceeding/FlowProblemParameters.cpp \
     ${OPM_ROOT}/opm-simulators/opm/simulators/flow/FlowProblemParameters.cpp
COPY runscripts/ECMOR24_proceeding/FlowProblemParameters.hpp \
     ${OPM_ROOT}/opm-simulators/opm/simulators/flow/FlowProblemParameters.hpp
COPY runscripts/ECMOR24_proceeding/MLNearWellConfig.hpp \
     ${OPM_ROOT}/opm-simulators/opm/simulators/wells/MLNearWellConfig.hpp
COPY runscripts/ECMOR24_proceeding/StandardWell.hpp \
     ${OPM_ROOT}/opm-simulators/opm/simulators/wells/StandardWell.hpp
COPY runscripts/ECMOR24_proceeding/StandardWell_impl.hpp \
     ${OPM_ROOT}/opm-simulators/opm/simulators/wells/StandardWell_impl.hpp

# Build OPM components.
RUN cmake -S ${OPM_ROOT}/opm-common -B ${OPM_BUILD}/opm-common \
 && cmake --build ${OPM_BUILD}/opm-common -j ${OPM_BUILD_JOBS}

RUN cmake -S ${OPM_ROOT}/opm-grid -B ${OPM_BUILD}/opm-grid \
 -DCMAKE_PREFIX_PATH=${DUNE_INSTALL} \
 -Ddune-grid_DIR=${DUNE_INSTALL}/lib/cmake/dune-grid \
 && cmake --build ${OPM_BUILD}/opm-grid -j ${OPM_BUILD_JOBS}

RUN mkdir -p ${OPM_BUILD}/opm-simulators && \
 cd ${OPM_BUILD}/opm-simulators && \
 cmake -DCMAKE_BUILD_TYPE=Release ${OPM_ROOT}/opm-simulators \
 && make -j ${OPM_BUILD_JOBS} flow_gaswater_dissolution_diffuse

RUN cmake -S ${OPM_ROOT}/opm-upscaling -B ${OPM_BUILD}/opm-upscaling \
 && cmake --build ${OPM_BUILD}/opm-upscaling -j ${OPM_BUILD_JOBS}

# Remove source files.
RUN for repo in opm-common opm-grid opm-simulators opm-upscaling; do \
      rm -rf ${OPM_ROOT}/${repo}; \
    done

# Ensure standard install locations and source-built OPM Flow are on PATH.
ENV LD_LIBRARY_PATH=/opt/dune/install/lib
ENV PATH="/usr/local/bin:${PATH}"

RUN ln -s /opt/opm_build/opm-simulators/bin/flow_gaswater_dissolution_diffuse /usr/local/bin/flow \
 && ln -s /opt/opm_build/opm-common/bin/co2brinepvt /usr/local/bin/co2brinepvt

# Create a non-root user to run the reproducibility workflow.
RUN useradd -ms /bin/bash diligent_researcher
USER diligent_researcher
ENV HOME=/home/diligent_researcher
WORKDIR ${HOME}

RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Copy ML_near_well repository into the image and install Python dependencies.
COPY --chown=diligent_researcher:diligent_researcher . ./ML_near_well
RUN cd ML_near_well && \
    python3.10 -m pip install --no-cache-dir -r requirements.txt

# Clone and install pyopmnearwell.
RUN git clone --branch 2024-08_ML_near_well_article \
      https://github.com/cssr-tools/pyopmnearwell && \
    cd pyopmnearwell && \
    python3.10 -m pip install --no-cache-dir -e .

CMD ["bash", "-lc", "sleep infinity"]
# Run the reproducibility workflow.
#CMD ["bash", "-lc", "cd ./ML_near_well && bash runscripts/run.bash"]