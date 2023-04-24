##########################################################################################
# Dependent tags
##########################################################################################

ARG ACT3_OCI_REGISTRY

##########################################################################################
# Dependent images
##########################################################################################

FROM ${ACT3_OCI_REGISTRY}/act3-rl/corl/development/user-base-builder:v1.52.6  as user_base_builder_base

#########################################################################################
# develop stage contains base requirements. Used as base for all other stages
#########################################################################################

ARG IMAGE_REPO_BASE
FROM ${IMAGE_REPO_BASE}docker.io/python:3.8 as develop
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore
ARG PIP_INDEX_URL
ARG APT_MIRROR_URL
ARG SECURITY_MIRROR_URL
ARG ACT3_TOKEN
ARG GIT_URL

#Sets up apt mirrors to replace the default registries
RUN if [ -n "$APT_MIRROR_URL" ] ; then sed -i "s|http://archive.ubuntu.com|${APT_MIRROR_URL}|g" /etc/apt/sources.list ; fi && \
if [ -n "$SECURITY_MIRROR_URL" ] ; then sed -i "s|http://security.ubuntu.com|${SECURITY_MIRROR_URL}|g" /etc/apt/sources.list ; fi


# retrieve sas repo and use install script to set up dependencies

ENV SA_SIMS_ROOT=/opt/safe-autonomy-sims

WORKDIR ${SA_SIMS_ROOT}

COPY . .

RUN chmod +x install.sh
RUN ./install.sh -p ${ACT3_TOKEN} -u ${GIT_URL} -d


#########################################################################################
# build stage packages the source code
#########################################################################################

FROM develop as build
ENV SA_SIMS_ROOT=/opt/libact3-sa-sims

WORKDIR /opt/project
COPY . .

RUN python setup.py sdist bdist_wheel -d ${SA_SIMS_ROOT}

#########################################################################################
# package stage
#########################################################################################

# the package stage contains everything required to install the project from another container build
# NOTE: a kaniko issue prevents the source location from using a ENV variable. must hard code path

FROM scratch as package
COPY --from=build /opt/libact3-sa-sims /opt/libact3-sa-sims

#########################################################################################
# User Base builder container
#########################################################################################
FROM user_base_builder_base as user-base-builder

ARG NEW_USER=act3rl
USER root
RUN chmod 1777 /tmp
ENV CODE=/opt/project/
RUN rm -rf $CODE/*

COPY --chown=${NEW_USER} --from=develop /opt/lib* /opt/

COPY . .

RUN chmod +x install.sh
RUN ./install.sh -p ${ACT3_TOKEN} -d

USER ${NEW_USER}

COPY --chown=${NEW_USER} . $CODE

#########################################################################################
# CI/CD stages. DO NOT make any stages after cicd
#########################################################################################

# the CI/CD pipeline uses the last stage by default so set your stage for CI/CD here with FROM your_ci_cd_stage as cicd
# this image should be able to run and test your source code
# python CI/CD jobs assume a python executable will be in the PATH to run all testing, documentation, etc.

FROM develop as cicd
