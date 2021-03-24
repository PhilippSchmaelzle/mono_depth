import tensorflow as tf

meta_path = '/home/fascar/Documents/mono_depth_training/models/backup_cityscape_416x128/100000_plus_25000_kofif_test_to_freeze/model.ckpt-125000.meta'
output_node_names = ['output:0']    # Output nodes

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess,tf.train.latest_checkpoint('/home/fascar/Documents/mono_depth_training/models/backup_cityscape_416x128/100000_plus_25000_kofif_test_to_freeze/'))

    # Freeze the graph
    output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    print(output_node_names)
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    #with open('output_graph.pb', 'wb') as f:
    #  f.write(frozen_graph_def.SerializeToString())