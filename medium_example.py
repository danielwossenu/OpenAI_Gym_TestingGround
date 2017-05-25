#code adapted from https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import gym

env = gym.make('CartPole-v0')

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1,4],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([4,2],0,0.01))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1,2],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)


init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 1000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        print i
        #Reset environment and get first new observation
        s = env.reset().flatten()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 99:
            env.render()
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            # a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(4)[s:s+1]})
            a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.array([s])})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])
            s1 = s1.flatten()
            #Obtain the Q' values by feeding the new state through our network
            # Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(4)[s1:s1+1]})
            Q1 = sess.run(Qout, feed_dict={inputs1: np.array([s1])})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.array([s]),nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                # print rAll
                break
        jList.append(j)
        rList.append(rAll)
print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"

plt.plot(rList)
plt.show()

