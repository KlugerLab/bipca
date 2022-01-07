# Use nvidia/cuda image
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG UNAME
ARG UID
ARG GID

WORKDIR /home/$UNAME/proj
# set bash as current shell
SHELL ["/bin/bash", "-c"]
RUN whoami

RUN apt-get update && \
	apt-get -y install sudo
# I think the following code sets the user
RUN echo "UNAME: $UNAME, UID: $UID, GID: $GID"
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME
RUN usermod -aG sudo $UNAME
RUN sudo chown -R $UNAME:$UNAME /home/$UNAME/
RUN sudo passwd -d $UNAME
RUN whoami
USER $UNAME
RUN whoami

# install anaconda
RUN sudo apt-get update -y && \
        sudo apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        sudo apt-get clean

RUN wget -O \
        mambaforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
        && bash mambaforge.sh -b \
        && source /home/jay/mambaforge/bin/activate
RUN ls -al
RUN echo $PATH
