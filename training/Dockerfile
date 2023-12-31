FROM tensorflow/tensorflow:latest-py3

LABEL maintainer="Shuhei Iitsuka <tushuhei@google.com>"

RUN apt-get update && \
    apt-get install -y git \
        python3 \
        python3-pip \
        apt-utils && \
    apt-get clean

RUN pip3 install -U pip

RUN pip3 install tensorflow==1.14

RUN pip3 --no-cache-dir install tensorflowjs==1.2.6

WORKDIR /

RUN git clone --depth 1 https://github.com/tensorflow/tensorflow.git --branch r1.7

CMD python3 /tensorflow/tensorflow/examples/image_retraining/retrain.py \
        --image_dir /data/images \
        --how_many_training_steps=4000 \
        --eval_step_interval=100 \
        --architecture mobilenet_0.25_224 \
        --output_graph /data/graph.pb \
        --summaries_dir /data/summaries \
        --output_labels /data/output_labels.txt \
        --bottleneck_dir /data/bottleneck/ \
        --intermediate_store_frequency 1000 \
        --intermediate_output_graphs_dir /data/intermediate \
        --saved_model_dir /data/saved_model && \
    echo 'export const SCAVENGER_CLASSES: {[key: number]: string} = {' > /data/scavenger_classes.ts && \
    awk '{print NR-1  ": '\''" $0 "'\'',"}' /data/output_labels.txt >> /data/scavenger_classes.ts && \
    echo '}' >> /data/scavenger_classes.ts && \
    python3 -m tensorflowjs.converters.converter \
        --input_format=tf_saved_model \
        --saved_model_tags=serve \
        /data/saved_model/ \
        /data/saved_model_web/
