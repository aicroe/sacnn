import './comment-input.js';

window.onload = function () {
  const classifier = new Vue({
    el: '#classifier',
    data() {
      return {
        newComment: { text: '' },
        comments: [],
        instanceName: null,
        errorText: null,
      };
    },
    methods: {
      classifyComment: async function (event) {
        event.preventDefault();
        if (!this.instanceName) {
          this.errorText = 'Debe seleccionar una instancia';
          return;
        }
        if (this.comments.length > 0) {
          document.body.style.cursor = 'wait';
          this.errorText = null;
          const response = await fetch('/classify', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              'comments': this.comments.map((comment) => comment.text),
              'instance_name': this.instanceName,
            }),
          });
          const data = await response.json();
          this.comments = this.comments.map((comment, index) => Object.assign({}, comment, {
            result: data.results[index]
          }));
          document.body.style.cursor = 'default';
        } else {
          this.errorText = 'No puede realizar la clasificación sino añade almenos un comentario';
        }
      },
      addNewComment() {
        if (this.newComment.text.length === 0) {
          this.errorText = 'No puede añadir un comentario vacio';
          return;
        }
        this.comments.unshift(this.newComment);
        this.newComment = { text: '' };
        this.errorText = null;
      },
      removeComment(index) {
        this.comments.splice(index, 1);
      },
      reset(event) {
        event.preventDefault();
        this.comments = [];
        this.newComment = { text: '' };
        this.errorText = null;
      }
    }
  });
}