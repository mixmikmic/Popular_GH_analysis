import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)  # tf.float32 implicitly
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))

logs_path = '/home/ubuntu/tensorflow-logs'
# make the directory if it does not exist
get_ipython().system('mkdir -p $logs_path')

# tensorboard --purge_orphaned_data --logdir /home/ubuntu/tensorf low-logs

summary_writer = tf.summary.FileWriter(
    logs_path, graph=tf.get_default_graph())

node3 = tf.add(node1, node2)
print('node3: ', node3)
print('sess.run(node3): ', sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3.0
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

summary_writer.close()

sess.close()

