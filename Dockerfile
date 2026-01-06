FROM continuumio/miniconda3

# Set the working directory
WORKDIR /opt/project

# Copy the environment.yml file and create the conda environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Copy user-editable weights config into the image
COPY weights.json .

# Activate the conda environment
SHELL ["conda", "run", "-n", "optimizer", "/bin/bash", "-c"]

# Copy the source code into the container and set the PATH
COPY src/ ./src/
ENV PATH="/opt/project/src:${PATH}"

# Activate the conda environment in the entrypoint script
COPY src/utils/entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]