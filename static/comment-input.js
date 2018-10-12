Vue.component('comment-input', {
  props: ['comment', 'index', 'actionLabel', 'enabled', 'error'],
  template: `
    <div class="inline fields">
      <div
        v-if="!!comment.result"
        class="two wide field"
      >
        <h2 class="result">{{comment.result}}</h2>
      </div>
      <div
        class="fourteen wide field"
      >
        <textarea
          rows="2"
          v-model="comment.text"
          placeholder="Comentario..."
          v-bind:disabled="!enabled"
          v-bind:class="{ 'error border': error }"
          required
        ></textarea>
      </div>
      <div 
        v-if="!comment.result"
        class="two wide field centered"
      >
        <button
          class="ui secondary button font large"
          v-on:click="action"
        >{{actionLabel}}</button>
      </div>
    </div>
  `,
  methods: {
    action(event) {
      event.preventDefault();
      this.$emit('action', this.index);
    }
  }
});