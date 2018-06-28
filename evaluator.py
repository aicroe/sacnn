from process_data import create_1vec_labels
from process_data import process_comments
from process_data import load_embedding
from model import Parameters, HyperParameters, SACNNBase, DataManager
import numpy as np
import tensorflow as tf

sentence_length = 100
word_dimension = 300
channels = 1
EMBEDDING_PATH = 'raw/SBW-vectors-300-min5.bin'
embedding = load_embedding(EMBEDDING_PATH)
input_data = tf.placeholder(tf.float32, shape=[None, sentence_length, word_dimension, channels])

(layer1_list_filters,
layer1_list_biases,
layer2_weights,
layer2_biases,
layer3_weights,
layer3_biases) = DataManager.load_parameters(layer1_params_expected=3)

parameters = Parameters(
    map(lambda layer1_filters: tf.constant(layer1_filters), layer1_list_filters),
    map(lambda layer1_biases: tf.constant(layer1_biases), layer1_list_biases),
    tf.constant(layer2_weights),
    tf.constant(layer2_biases),
    tf.constant(layer3_weights),
    tf.constant(layer3_biases))

hparameters = HyperParameters(learning_rate=0)

model = SACNNBase(
    parameters,
    hparameters,
    input_data,
    None,
    keep_prob=1)

prediction = model.prediction

def evaluate_comment(comment):
    comment_matrix = process_comments(
        embedding.wv, np.array([comment]), 
        1, 
        sentence_length, 
        word_dimension, 
        channels)
    with tf.Session() as session:
        return np.argmax(session.run(prediction, feed_dict={input_data: comment_matrix})) + 1

if __name__ == '__main__':
    print(evaluate_comment('Buen Hotel en una ciudad bonita. La ubicación no llega a ser de las mejores pero tiene otras muchas ventajas. La calidad de las habitaciones, los detalles cuidados y la atención del personal son los puntos a destacar. El precio lo encontré muy bueno para un Hotel de esta gama.'))
    print(evaluate_comment('Muy buen lugar para una estadía tranquila dado su lejanía del centro d Cochabamba. Hotel limpio y de habitaciones amplias. Personal agradable y atento. Inconvenientes con equipos de aire acondicionado.'))
    print(evaluate_comment('''Aunque las opiniones parecieran positivas... Mi percepción es bien distinta.

Lo positivo: agua caliente y con presión.. Habitaciones modernas y limpias.

Lo negativo: toallas muy pequeñas. La cama.. El colchón con las muellas reventadas... De quien viene en un biaje de trabajo... Q llega tarde y agotado.. No es nada placentero.

Las habitación estándar NO tienen secador de pelo. En la recepción no presentan alternativa ni tienen para prestar. :(

Desayuno: buffet básico y si estoy de acuerdo con la opinión de enero que acá hicieran... De cobrar por unos huevos revueltos... Cuando tienen huevos duros... Es decir... Aunque la gerencia dice que lo tendría en consideración... 9 meses después... Nada cambio.

Hay detalles fáciles que pueden cambiar y que dejarían el huésped mucho mas agradado'''))
    print(evaluate_comment('Nos quedamos en una habitación doble, el baño muy malo, la puerta de la ducha estaba rota, apenas se podía cerrar, el desagüe estaba malo, lo que hacia que la pieza tuviera un olor desagradable, además que la limpieza del baño no era muy buena. Las camas horribles, muy incomodas, el wifi solo funciona en recepción y a veces. En Bolivia los enchufes son de 220, y en este hostal tenían todos adaptados a 110, no teníamos como cargar los celulares mas que en recepción. Nos tuvimos que cambiar de hostal finalmente. Ademas los precios son muy caros comparados con otras hostales.'))
    print(evaluate_comment('Reservé una habitación con cama de matrimonio y me dieron una doble porque no tenían. Los baños y duchas estan hechos una porquería, con suerte te sale agua caliente, yo no lo probé en tres noches. Las habitaciones son frías, el wifi es inexistente, excepto en la recepción. Eso si, para los que os guste el flocklore, cada día de 20.30 a 22.30 en la primera planta tenéis ensayo. Perfecto para no descansar mientras el edificio tiembla. De verdad que se paga mucho mas de lo que se merece.'))