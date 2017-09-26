import tensorflow as tf

#download and read the data automatically
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#note the one_hot = True --> one hot vector
#vectors which is 0 in most dimensions and 1 in a single dim
#nth dig = vectors which is 1 in nth dimension - for this case

#start building the model

#3 hidden layers with 500 nodes each
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#
n_classes = 10

#batch_size = num of samples that are going to be propagate thorugh
#the network - therefore alg in this cse takes
#the first 100 samples from training dataset
#then the second 100 samples etc
#-- good cause update weights after each propagation
batch_size = 100

#placeholders for values in graph
#n.b. we create the MODEL in tf graph
#then tf manipulates everything
x = tf.placeholder('float', [None, 784]) #the dimensions can be any length, 784-dim vector
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}


    #feed forward

    #matmul = multiplication matrices
    #multiplication of raw data and their weights and then adding bias
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])

    #this is the activation function
    #return 0 if value negative
    #else return the value itself if 0 or +ve
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)


    #outputs chance of a data to be in any of the 10 classes
    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']
    #not so sure why this is added differently...

    return output


#training process
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    #cost variable = how wrong we are

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    #optimizer of cost function - popular along stoch gradient descent

    hm_epochs = 10
    #determines number of epochs to have
    #epoch = cycles of feed forward and back prop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #initialize all vars

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y}) #feed_dict gives values to the placeholders
                #note that this run returns an array with the two values, optimizer and cost
                epoch_loss += c

            print ('Epoch', epoch, 'completed out of', hm_epochs, 'loss: ', epoch_loss)


    #for each epoch and for each batch in our data
    #we run the optimizer and cost against our batch
    #to keep track of our loss/cost we add tot cost per epoch up
    #for each epoch we output the loss - shoudl decrease all the time

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # tells us number of predictions that were matches to labels
        #gives index of highest entry in a tensor along some axis

        #gives a list of booleands therefore have to cast it as array integers then take the mean

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)




#tf.random_noermal - outputs random values for the shape we want
#i.e. for the shape that the layer's matrix should be
#shape is given by stating the size of matrix [     ]
