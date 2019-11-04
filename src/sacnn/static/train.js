window.onload = function () {
  const status = {
    idle: 'IDLE',
    training: 'TRAINING',
    trainCompleted: 'TRAIN_COMPLETED',
    trainFailed: 'TRAIN_FAILED',
  };
  new Vue({
    el: '#trainer',
    data: {
      arch: 'kim',
      instanceName: '',
      trainer: 'simple',
      hiddenUnits: 64,
      numLabels: 3,
      learningRate: 0.009,
      epochs: 100,
      validationGap: 5,
      minibatchSize: 32,
      keepProb: 0.5,
      trainStatus: '',
      error: '',
    },
    methods: {
      trainInstance
    }
  });

  async function trainInstance(event) {
    this.instanceName = this.instanceName.trim();
    if (this.instanceName.length > 0 &&
      this.hiddenUnits > 0 &&
      (this.numLabels === 3 || this.numLabels === 5) &&
      this.learningRate > 0 &&
      this.epochs > 0 &&
      this.validationGap > 0 &&
      this.minibatchSize > 0 &&
      this.keepProb > 0) {

      event.preventDefault();
      document.body.style.cursor = 'wait';

      const response = await fetch('/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          arch: this.arch,
          name: this.instanceName,
          trainer: this.trainer,
          hidden_units: +this.hiddenUnits,
          num_labels: +this.numLabels,
          learning_rate: +this.learningRate,
          epochs: +this.epochs,
          validation_gap: +this.validationGap,
          minibatch_size: +this.minibatchSize,
          keep_prob: +this.keepProb
        })
      });
      if (response.status === 200) {
        this.trainStatus = status.idle;
        startTraining.call(this, this.instanceName, this.epochs);
      } else {
        this.error = response.statusText;
      }
      document.body.style.cursor = 'default';
    }
  }

  function startTraining(instanceName, epochs) {
    const progressBar = $('#progress-bar')
      .progress({
        total: epochs,
        text: {
          active: 'Entrenando. Iteración {value} de {total}',
          success: 'Entrenamiento completado!'
        }
      });
    progressBar.progress('set progress', 0);
    const interval = setInterval(async () => {
      let state;
      try {
        state = await askTrainingState(instanceName);
      } catch (error) {
        this.error = error || 'UNEXPECTED ERROR';
        return clearInterval(interval);
      }
      this.trainStatus = state['status'];
      if (this.trainStatus === status.trainFailed) {
        this.error = this.trainStatus;
        clearInterval(interval);
      } else if (this.trainStatus === status.trainCompleted) {
        progressBar.progress('set progress', epochs);
        clearInterval(interval);
      } else {
        progressBar.progress('set progress', +state['epoch']);
      }
    }, 5000);
  }

  async function askTrainingState(instanceName) {
    const response = await fetch('/train_state', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        name: instanceName
      })
    });
    if (response.status === 200) {
      return response.json();
    } else {
      throw response.statusText;
    }
  }
}