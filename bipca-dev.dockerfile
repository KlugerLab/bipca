# Use nvidia/cuda image
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG UNAME
ARG UID
ARG GID
ARG ENV=*.yml
WORKDIR /home/$UNAME/container
COPY $ENV ./environment.yml
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

VOLUME /code
# install anaconda
RUN sudo apt-get update -y && \
        sudo apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        sudo apt-get clean

RUN wget -O \
        mambaforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
        && bash mambaforge.sh -b \
        && source /home/${UNAME}/mambaforge/bin/activate
ENV PATH="/home/$UNAME/mambaforge/bin:$PATH"
ARG PATH="/home/$UNAME/mambaforge/bin:$PATH"

RUN mamba env create -f environment.yml
RUN mamba init
RUN mamba install jupyterlab
RUN jupyter-lab --generate-config
RUN ipython profile create
RUN printf "\nc.ServerApp.ip = '0.0.0.0'\n" >> /home/$UNAME/.jupyter/jupyter_lab_config.py
RUN printf "\nc.ServerApp.token = ''\n" >> /home/$UNAME/.jupyter/jupyter_lab_config.py
RUN printf "\nc.ServerApp.password = ''\n" >> /home/$UNAME/.jupyter/jupyter_lab_config.py
RUN printf "\nc.ServerApp.port = 8080\n" >> /home/$UNAME/.jupyter/jupyter_lab_config.py
RUN printf "\nc.NotebookApp.notebook_dir = '/home'\n" >> /home/$UNAME/.jupyter/jupyter_lab_config.py
RUN printf "\nc.NotebookApp.terminado_settings={'shell_command':['bash', '--login']}" >>/home/$UNAME/.jupyter/jupyter_lab_config.py

RUN printf "\nc = get_config()" >> /home/$UNAME/.ipython/profile_default/ipython_config.py
RUN printf "\nc.InlineBackend.print_figure_kwargs={'facecolor' : 'w'}">> /home/$UNAME/.ipython/profile_default/ipython_config.py
RUN printf "\nc = get_config()" >> /home/$UNAME/.ipython/profile_default/ipython_kernel_config.py
RUN printf "\nc.InlineBackend.print_figure_kwargs={'facecolor' : 'w'}">> /home/$UNAME/.ipython/profile_default/ipython_kernel_config.py
RUN echo "conda activate bipca-experiment" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
#COPY entrypoint.sh ./
#RUN chown -R $UNAME:$UNAME entrypoint.sh
#RUN chmod +x entrypoint.sh
#ENTRYPOINT ["./entrypoint.sh"]

ENTRYPOINT ["conda","run","--no-capture-output","-n","bipca-experiment","pip","install","-e","/code","&","conda","run","-n","bipca-experiment","jupyter-lab","&","/bin/bash"]
