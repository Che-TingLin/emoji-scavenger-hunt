import * as tf from '@tensorflow/tfjs';
import {SCAVENGER_CLASSES} from './scavenger_classes';

const MODEL_URL = '/model/model.json';
const PREPROCESS_DIVISOR = tf.scalar(255 / 2);

export class MobileNet {
  model: tf.GraphModel;

  async load() {
    this.model = await tf.loadGraphModel(MODEL_URL);
  }

  dispose() {
    if (this.model) {
      this.model.dispose();
    }
  }

  predict(input: tf.Tensor): tf.Tensor1D {
    const preprocessedInput = tf.div(
        tf.sub(input.asType('float32'), PREPROCESS_DIVISOR),
        PREPROCESS_DIVISOR);
    const reshapedInput = preprocessedInput.reshape([1, ...preprocessedInput.shape]);
    return this.model.predict(reshapedInput) as tf.Tensor1D;
  }

  getTopKClasses(predictions: tf.Tensor1D, topK: number) {
    const values = predictions.dataSync();
    predictions.dispose();

    let predictionList = [];
    for (let i = 0; i < values.length; i++) {
      predictionList.push({value: values[i], index: i});
    }
    predictionList = predictionList.sort((a, b) => {
      return b.value - a.value;
    }).slice(0, topK);

    return predictionList.map(x => {
      return {label: SCAVENGER_CLASSES[x.index], value: x.value};
    });
  }
}
