



import numpy as np
import tensorflow as tf

def DQNEval(usableAcePlayer, playerSum, dealerCard1,ANN):
    phi=(usableAcePlayer, playerSum, dealerCard1)
    uh=ANN.get_action_scores(phi)
    return np.reshape(uh, (2,))

def DQNTrain(usableAcePlayer, playerSum, dealerCard1, action, reward,ANN):
    ANN.buffer[0].append([usableAcePlayer, playerSum, dealerCard1])
    if reward>=0:
        rl_action = np.zeros(2)
        rl_action[action]= 1
    elif reward<0:
        rl_action = np.ones(2)
        rl_action[action]= 0

    ANN.buffer[1].append(rl_action)
    if len(ANN.buffer[0]) >= 50:
        ANN.trainBatch(ANN.buffer)
        ANN.buffer = [[], []]


def getAction(usableAcePlayer, playerSum, dealerHand, actions,explorprob,ANN):
    if np.random.random(1) < explorprob:
        uh1=int(0+(2-0)*np.random.random(1))
    else:
        uh2=DQNEval(usableAcePlayer, playerSum, dealerHand,ANN)
        if max(uh2)==min(uh2):
            uh1=None
        else:
            uh1=actions[np.argmax(uh2)]
    return uh1


class Qnet():
    def __init__(self):
        self.buffer = [[], []]
        self.n_hidden1 = 40
        self.n_hidden2 = 20

        self.x = tf.placeholder(tf.float32, shape=[None, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])

        self.W_fc1 = tf.get_variable('W_fc1', shape=[3,self.n_hidden1])
        self.b_fc1 = tf.get_variable('b_fc1', shape=[self.n_hidden1])
        self.h_fc1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.W_fc1), self.b_fc1, name='layer1'))

        self.W_fc2 = tf.get_variable('W_fc2', shape=[self.n_hidden1,self.n_hidden2])
        self.b_fc2 = tf.get_variable('b_fc2', shape=[self.n_hidden2])
        self.h_fc2 = tf.nn.relu(tf.add(tf.matmul(self.h_fc1, self.W_fc2), self.b_fc2, name='layer2'))

        self.W_fc3 = tf.get_variable('W_fc3', shape=[self.n_hidden2,2])
        self.b_fc3 = tf.get_variable('b_fc3', shape=[2])
        self.h_fc3 = tf.add(tf.matmul(self.h_fc2, self.W_fc3), self.b_fc3, name='layer3')

        self.q = tf.nn.softmax(self.h_fc3)

        self.loss = tf.reduce_sum(tf.square(self.y - self.q),1)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver = tf.train.Saver()

    def get_action_scores(self, x):
        with self.sess.as_default():
            return self.q.eval(feed_dict={self.x: np.reshape(x, [-1,3])})

    def trainBatch(self, batch):
        xx = np.array(batch[0])
        yy = np.array(batch[1])
        with self.sess.as_default():
            self.sess.run(self.train_step, feed_dict={
                                                        self.x: xx,
                                                        self.y: yy,
                                                        })


def SumHand(Hand):
    if Hand[0]==1 and Hand[1]!=1:
        totalsum=11+Hand[1]
        usableAce=1
    elif Hand[0]!=1 and Hand[1]==1:
        totalsum=11+Hand[0]
        usableAce=1
    elif Hand[0]==1 and Hand[1]==1:
        totalsum=11+1
        usableAce=1
    else:
        totalsum=Hand[0]+Hand[1]
        usableAce=0
    return totalsum,usableAce

def mapa(actions,ANN):
    print("\n\n\n\n")
    print("Player without usable Ace\n")
    print("0 = stand || 1 = hit")
    print(" ---------------------- <- Player shows")
    print("|\t2 3 4 5 6 7 8 9 T A <- Dealer shows")
    print("v")
    usableAcePlayer=0
    action0=0
    action1=1
    for playerSum in range(1,22):
        vv=[]
        for dealerCard1 in range(1,11):
            action = getAction(usableAcePlayer, playerSum, dealerCard1,actions,0,ANN)

            if action is None:
                uhh='?'
            elif action==1:
                uhh=0
            elif action==0:
                uhh=1
            vv.append(uhh)

        print("{}\t{} {} {} {} {} {} {} {} {} {}".format(playerSum,vv[1],vv[2],vv[3],vv[4],vv[5],vv[6],vv[7],vv[8],vv[9],vv[0]))



    print("\nPlayer with usable Ace\n")
    print("0 = stand || 1 = hit")
    print(" ---------------------- <- Player shows")
    print("|\t2 3 4 5 6 7 8 9 T A <- Dealer shows")
    print("v")
    usableAcePlayer=1
    action0=0
    action1=1
    for playerSum in range(1,22):
        vv=[]
        for dealerCard1 in range(1,11):
            action = getAction(usableAcePlayer, playerSum, dealerCard1,actions,0,ANN)

            if action is None:
                uhh='?'
            elif action==1:
                uhh=0
            elif action==0:
                uhh=1

            vv.append(uhh)
            uh=0
        uh=0
        print("{}\t{} {} {} {} {} {} {} {} {} {}".format(playerSum,vv[1],vv[2],vv[3],vv[4],vv[5],vv[6],vv[7],vv[8],vv[9],vv[0]))



def getCard():
    return min(int(1+14*np.random.random(1)), 10)


def play_blackjack(playerHand,dealerHand,actions,explorprob,hit,stand,policyDealer,ANN):

    playerSum,usableAcePlayer=SumHand(playerHand)
    dealerSum,usableAceDealer=SumHand(dealerHand)

    playerTrajectory=[]
    reward=0
    salida=0
    while salida==0:
        action = getAction(usableAcePlayer, playerSum, dealerHand[0],actions,explorprob,ANN)
        if action is None:
            action=random.choice(actions)
        playerTrajectory.append([action,(usableAcePlayer,playerSum,dealerHand[0])])

        if action==stand:
            salida=1
        else:
            playerSum=playerSum+getCard()
            if playerSum>21:
                if usableAcePlayer==1:
                    playerSum=playerSum-10
                    usableAcePlayer=0
                else:
                    reward=-1
                    salida=1
    if reward==0:
        salida=0
        while salida==0:
            action=int(policyDealer[dealerSum])
            if action==stand:
                salida=1
            else:
                dealerSum=dealerSum+getCard()
                if dealerSum>21:
                    if usableAceDealer==1:
                        dealerSum=dealerSum-10
                        usableAceDealer=0
                    else:
                        reward=1
                        salida=1
    if reward==0:
        if playerSum>dealerSum:
            reward=1
        elif playerSum==dealerSum:
            reward=0
        else:
            reward=-1
    return reward,playerTrajectory



def train(numsim,actions,hit,stand,batch_size,ANN):

    total_reward = 0.0
    winnings = 0.0


    policyDealer = np.zeros(22)
    for i in range(17, 22):
        policyDealer[i] = stand

    for iter in range(numsim):

        if iter % batch_size == 0:
            print("from iter {} to iter {}, reward: {}".format(max(iter-batch_size,0),iter, total_reward))
            winnings += total_reward
            total_reward = 0.0


        playerHand = []
        dealerHand = []

        for _ in range(2):
            playerHand.append(getCard())
            dealerHand.append(getCard())

        reward,playerTrajectory=play_blackjack(playerHand,dealerHand,actions,0.2,hit,stand,policyDealer,ANN)
        total_reward += reward
        for cont in range(10):
            for action, (usableAcePlayer, playerSum, dealerCard1) in playerTrajectory:
                DQNTrain(usableAcePlayer, playerSum, dealerCard1, action, reward,ANN)


def test(numsim,actions,hit,stand,ANN):

    total_reward = 0.0

    policyDealer = np.zeros(22)
    for i in range(17, 22):
        policyDealer[i] = stand

    for iter in range(numsim):
        playerHand = []
        dealerHand = []

        for _ in range(2):
            playerHand.append(getCard())
            dealerHand.append(getCard())

        reward,playerTrajectory=play_blackjack(playerHand,dealerHand,actions,0,hit,stand,policyDealer,ANN)

        total_reward += reward
    print("\nE[reward]: {}".format(total_reward/numsim))


def main():

    np.random.seed(314)

    hit=0
    stand=1
    actions=[hit,stand]
    batch_size = 10000
    num_sim=1000000

    ANN=Qnet()

    train(num_sim,actions,hit,stand,batch_size,ANN)
    test(batch_size,actions,hit,stand,ANN)
    mapa(actions,ANN)



if __name__ == '__main__': main()
