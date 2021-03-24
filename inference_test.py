import tensorflow as tf

class CNN(object):

    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath = self.model_filepath)

    def load_graph(self, model_filepath):
        '''
        Lode trained model.
        '''
        print('Loading model...')
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph = self.graph)

        with tf.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        print('Check out the input placeholders:')
        nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
        for node in nodes:
            print(node)

        # Define input tensor
        self.input = tf.placeholder(np.float32, shape = [None, 32, 32, 3], name='input')
        self.dropout_rate = tf.placeholder(tf.float32, shape = [], name = 'dropout_rate')

        tf.import_graph_def(graph_def, {'input': self.input, 'dropout_rate': self.dropout_rate})

        print('Model loading complete!')

        '''
        # Get layer names
        layers = [op.name for op in self.graph.get_operations()]
        for layer in layers:
            print(layer)
        '''

        '''
        # Check out the weights of the nodes
        weight_nodes = [n for n in graph_def.node if n.op == 'Const']
        for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            print("Value - " )
            print(tensor_util.MakeNdarray(n.attr['value'].tensor))
        '''

    def test(self, data):

        # Know your output node name
        output_tensor = self.graph.get_tensor_by_name("import/cnn/output:0")
        output = self.sess.run(output_tensor, feed_dict = {self.input: data, self.dropout_rate: 0})

        return output




detection_graph = tf.Graph()

graph_path = '/home/fascar/Documents/mono_depth_training/models/backup_cityscape_416x128/100000_plus_25000_kofif_test_to_freeze/output_graph.pb'

with detection_graph.as_default():
    object_detection_graph_def = tf.compat.v1.GraphDef()

    with tf.io.gfile.GFile(graph_path, 'rb') as fid:
        serialized_graph = fid.read()
        object_detection_graph_def.ParseFromString(
            serialized_graph)

        # fix nodes
        for node in object_detection_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
 

        tf.import_graph_def(object_detection_graph_def, name='')