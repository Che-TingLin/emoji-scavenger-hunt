# 使用 TensorFlow 2.6 的 Python 3 版本作為基礎映像
FROM tensorflow/tensorflow:2.6.0

LABEL maintainer="Shuhei Iitsuka <tushuhei@google.com>"

# 更新並安裝必要的軟件包
RUN apt-get update && \
    apt-get install -y git \
    python3 \
    python3-pip \
    apt-utils && \
    apt-get clean

# 更新 pip
RUN pip3 install -U pip

# 安裝 TensorFlow 2.6.0
RUN pip3 install tensorflow==2.6.0

RUN pip3 install --upgrade tensorflow-estimator==2.6.0

RUN pip3 install --upgrade keras==2.6.0

# 安裝 TensorFlow.js 3.11.0（與 tfjs 4.11.0 相匹配的版本）
RUN pip3 --no-cache-dir install tensorflowjs==3.11.0

# 安裝 tensorflow-hub 0.12.0 和 make_image_classifier
RUN pip3 install tensorflow-hub==0.12.0 "tensorflow-hub[make_image_classifier]"

WORKDIR /

# 清除先前的模型數據
CMD rm -rf /data/saved_model/ /data/saved_model_web/ && \
    # 使用 make_image_classifier 進行訓練
    make_image_classifier \
    --image_dir /data/images \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/5 \
    --image_size 224 \
    --saved_model_dir /data/saved_model \
    --labels_output_file /data/output_labels.txt \
    --tflite_output_file /data/model.tflite \
    --summaries_dir /data/summaries \
    --train_epochs=10 && \
    echo 'export const SCAVENGER_CLASSES: {[key: number]: string} = {' > /data/scavenger_classes.ts && \
    awk '{print NR-1  ": '\''" $0 "'\'',"}' /data/output_labels.txt >> /data/scavenger_classes.ts && \
    echo '}' >> /data/scavenger_classes.ts && \
    python3 -m tensorflowjs.converters.converter \
    --input_format=tf_saved_model \
    --saved_model_tags=serve \
    /data/saved_model/ \
    /data/saved_model_web/
