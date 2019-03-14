window.onload = function () {
  const trainer = new Vue({
    el: '#trainer',
    data: {
      arch: 'kim',
      instanceName: '',
      iterator: 'simple',
      hiddenUnits: 64,
      numLabels: 3,
      learningRate: 0.009,
      epochs: 100,
      epochPrintCost: 5,
      minibatchSize: 32,
      keepProb: 0.5,
      trainState: '',
      error: '',
    },
    methods: {
      trainInstance
    }
  });

  async function trainInstance(event) {
    this.instanceName = this.instanceName.trim();
    if ((this.arch === 'kim' || this.arch === 'evolved') &&
      this.instanceName.length > 0 &&
      this.hiddenUnits > 0 &&
      (this.numLabels === 3 || this.numLabels === 5) &&
      this.learningRate > 0 &&
      this.epochs > 0 &&
      this.epochPrintCost > 0 &&
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
          iterator: this.iterator,
          hidden_units: this.hiddenUnits,
          num_labels: this.numLabels,
          learning_rate: this.learningRate,
          epochs: this.epochs,
          epoch_print_cost: this.epochPrintCost,
          minibatch_size: this.minibatchSize,
          keep_prob: this.keepProb
        })
      });
      if (response.status === 200) {
        this.trainState = 'initializing';
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
      let data;
      try {
        data = await askTrainingState(instanceName);
      } catch (error) {
        this.error = error || 'UNEXPECTED ERROR';
        return clearInterval(interval);
      }
      this.trainState = data['state'];
      if (this.trainState === 'finished') {
        progressBar.progress('set progress', epochs);
        clearInterval(interval);
      } else {
        progressBar.progress('set progress', +data['epoch']);
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