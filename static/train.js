window.onload = function () {
  const trainer = new Vue({
    el: '#trainer',
    data: {
      arch: 'base',
      instanceName: '',
      hiddenUnits: 64,
      numLabels: 3,
      learningRate: 0.001,
      epochs: 100,
      minibatchSize: 16,
      keepProb: 0.5,
      loading: false
    },
    methods: {
      trainInstance: function (event) {
        this.instanceName = this.instanceName.trim();
        if ((this.arch === 'base' || this.arch === 'evolved') &&
            this.instanceName.length > 0 &&
            this.hiddenUnits > 0 &&
            (this.numLabels === 3 || this.numLabels === 5) &&
            this.learningRate > 0 &&
            this.epochs > 0 &&
            this.minibatchSize > 0 &&
            this.keepProb > 0) {
          event.preventDefault();
          this.loading = true;
          document.body.style.cursor = 'wait';
          fetch('/train', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                arch: this.arch,
                instance_name: this.instanceName,
                hidden_units: this.hiddenUnits,
                num_labels: this.numLabels,
                learning_rate: this.learningRate,
                epochs: this.epochs,
                minibatch_size: this.minibatchSize,
                keep_prob: this.keepProb
              })
            })
            .then(response => response.json())
            .then(data => console.log(data))
            .catch(error => console.error(error))
            .finally(() => document.body.style.cursor = 'default');
        }
      }
    }
  })
}