window.onload = function () {
  const trainer = new Vue({
    el: '#trainer',
    data: {
      arch: 'kim',
      instanceName: '',
      hiddenUnits: 64,
      numLabels: 3,
      learningRate: 0.001,
      epochs: 100,
      epochPrintCost: 5,
      minibatchSize: 16,
      keepProb: 0.5,
      trainState: '',
    },
    methods: {
      trainInstance
    }
  });

  function trainInstance(event) {
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

      fetch('/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          arch: this.arch,
          name: this.instanceName,
          hidden_units: this.hiddenUnits,
          num_labels: this.numLabels,
          learning_rate: this.learningRate,
          epochs: this.epochs,
          epoch_print_cost: this.epochPrintCost,
          minibatch_size: this.minibatchSize,
          keep_prob: this.keepProb
        })
      })
        .then(response => {
          if (response.status === 200) {
            this.trainState = 'initializing';
            startTraining.call(this, this.instanceName, this.epochs);
          } else {
            throw response.statusText;
          }
        })
        .catch(error => console.error(error))
        .finally(() => document.body.style.cursor = 'default');
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
    const interval = setInterval(() => {
      askTrainingState(instanceName)
        .then(data => {
          this.trainState = data['state'];
          if (this.trainState === 'finished') {
            progressBar.progress('set progress', epochs);
            clearInterval(interval);
          } else {
            progressBar.progress('set progress', +data['epoch']);
          }
        })
        .catch(error => console.error(error));
    }, 5000);
  }

  function askTrainingState(instanceName) {
    return fetch('/train_state', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        name: instanceName
      })
    }).then((response) => {
      if (response.status === 200) {
        return response.json();
      } else {
        throw response.statusText;
      }
    });
  }
}