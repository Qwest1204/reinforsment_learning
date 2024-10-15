FROM nvidia/cuda:12.6.1-runtime-ubuntu24.04

COPY .aws /root/.aws
COPY . NN
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0
RUN apt install curl wget zip -y && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && ./aws/install

WORKDIR /NN


RUN python3 -m pip install -r requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN aws s3 cp s3://656d3985-mldata/data/data_840.zip data_840.zip --endpoint-url https://s3.timeweb.cloud && unzip data_840.zip -d /NN/data/

# RUN python3 VAE/train.py
ENTRYPOINT ["python"]
CMD ["VAE/train.py"] 

RUN aws s3 cp VAE s3://656d3985-mldata/VAE/ --endpoint-url https://s3.timeweb.cloud --recursive

