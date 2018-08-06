Vue.component('classify-result', {
  props: ['result'],
  template: `
    <div>
      <span class="result text">Resultado:</span>
      <h1 class="result">{{result}}</h1>
    </div>
  `
});

window.onload = function () {
  const classifier = new Vue({
    el: '#classifier',
    data: {
      result: '',
      comment: '',
      instanceName: ''
    },
    methods: {
      classifyComment: function (event) {
        if (this.comment.length > 0 && this.instanceName.length > 0) {
          event.preventDefault();
          document.body.style.cursor = 'wait';
          fetch('/classify', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                'comment' : this.comment,
                'instance_name': this.instanceName
              })
            })
            .then(response => response.json())
            .then(data => this.result = data.result)
            .catch(error => console.error(error))
            .finally(() => document.body.style.cursor = 'default')
        }
      }
    }
  });
}