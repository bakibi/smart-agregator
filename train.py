import tensorflow as tf
import numpy as np
from SvnGaussiKernel import SvnGaussiKernel
from batch_algo import batch_algo
import man_data


MAX_SEQUENCE = 18
EPOCH = 500
BATCH_SIZE = 2000
NUMBER_OF_CLASSIFIER = 5

def main():

    #PREPARING DATA

    training_file='data/train1.csv.zip'
    features,labels =man_data.traiter_donnees(man_data.read_file(training_file))
    features,dictionnaire = man_data.transformer_features(features,MAX_SEQUENCE)
    features = np.asarray(features)
    labels = np.transpose(np.asarray(labels))
    #features = features[:BATCH_SIZE,:]
    print(len(features),features.shape)
    #labels = labels[:BATCH_SIZE]
    labelsT = labels[:,:BATCH_SIZE]
    featuresT = features[:BATCH_SIZE,:]




    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph)

        #model = SvnGaussiKernel(MAX_SEQUENCE=MAX_SEQUENCE,ALL_X=featuresT,ALL_Y=labelsT,BATCH_SIZE=BATCH_SIZE,NUMBER_OF_CATEGORIES=7)
        model1 = batch_algo(MAX_SEQUENCE=MAX_SEQUENCE,ALL_X=features,ALL_Y=labels,BATCH_SIZE=BATCH_SIZE,NUMBER_OF_CATEGORIES=7,NUMBER_OF_CLASSIFIER=NUMBER_OF_CLASSIFIER)
        init = tf.global_variables_initializer()
        sess.run(init)

        #model.train(sess=sess,epoch=EPOCH)
        model1.trainAllClassifers(sess=sess,EPOCH=EPOCH)
        builder = tf.saved_model.builder.SavedModelBuilder('./SavedModel')
        builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING])
        builder.save()

        training_file='data/accur.csv'
        features,labels =man_data.traiter_donnees(man_data.read_file(training_file,False))
        features,dictionna = man_data.transformer_features(features,MAX_SEQUENCE,True,dictionnaire)
        #features,labels = man_data.labels_choose(features,labels,1,2)
        features = np.asarray(features)
        labels = np.asarray(labels)

        if len(features)>BATCH_SIZE:
            rand_index = np.random.choice(len(features), size=100)
            test_in = features[rand_index]
            test_out = labels[rand_index]
        else :

            test_in = features
            test_out = labels
        a = [np.argmax(test_out[i]) for i in range(len(test_out))]
        print(np.asarray(a))
        #out = model.predict(sess=sess,grid=test_in)
        out2 = model1.getPredict(sess=sess,data=test_in)

        print(np.asarray(out2))

        ans = 0
        #ans = man_data.accuracy_output(test_out,out)
        ans2 = man_data.accuracy_output(test_out,out2)
        print('accuary final est :',ans,ans2)



main()
