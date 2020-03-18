import numpy as np
class rnn(object):
    
    def derivative(self,b,c,W,U,V,y1,y2,h1,h2):
        da1_db1=1
        da2_db1=0
        for t in range(1,3):

            dh1_da1 = 1-np.square(h1[t])
            dh2_da2 = 1-np.square(h2[t])
            dh1_db1 = dh1_da1*da1_db1
            dh2_db1 = dh2_da2*da2_db1
            da1_db1 = 1+ W[0][0]*dh1_db1 + W[0][1]*dh2_db1
            da2_db1 = W[1][0]*dh1_db1 +W[1][1]*dh2_db1




        dL_da1 = (( (2*y1-1)*(y1*y2)+y1 ) *(V[0][0]) + ((-2*(y1-0.5)*(y1*y2)-y1) * V[1][0]) ) * (1-np.square(h1[3]))

        dL_da2 = (( (2*y1-1)*(y1*y2)+y1 ) *(V[0][1]) + ((-2*(y1-0.5)*(y1*y2)-y1) * V[1][1]) ) * (1-np.square(h2[3]))


        #dL_da2 = (y1*V[0][1] + (y2-1)*V[1][1]) * (1-np.square(h2[3]))

        #dL_da1 = (y1*V[0][0] + (y2-1)*V[1][0]) * (1-np.square(h1[3]))
        dL_db1 = dL_da1 * da1_db1 + dL_da2 * da2_db1

        return dL_db1



    def derivative_b2(self,b,c,W,U,V,y1,y2,h1,h2):
        
        da1_db2=0
        da2_db2=1
        for t in range(1,3):

            dh1_da1 = 1-np.square(h1[t])
            dh2_da2 = 1-np.square(h2[t])
            dh1_db2 = dh1_da1*da1_db2
            dh2_db2 = dh2_da2*da2_db2
            da1_db2 =  W[0][0]*dh1_db2 + W[0][1]*dh2_db2
            da2_db2 = 1+W[1][0]*dh1_db2 +W[1][1]*dh2_db2

        dL_da1 = (( (2*y1-1)*(y1*y2)+y1 ) *(V[0][0]) + ((-2*(y1-0.5)*(y1*y2)-y1) * V[1][0]) ) * (1-np.square(h1[3]))

        dL_da2 = (( (2*y1-1)*(y1*y2)+y1 ) *(V[0][1]) + ((-2*(y1-0.5)*(y1*y2)-y1) * V[1][1]) ) * (1-np.square(h2[3]))


        #dL_da2 = (y1*V[0][1] + (y2-1)*V[1][1]) * (1-np.square(h2[3]))

        #dL_da1 = (y1*V[0][0] + (y2-1)*V[1][0]) * (1-np.square(h1[3]))
        dL_db2 = dL_da1 * da1_db2 + dL_da2 * da2_db2

        return dL_db2



         
    def softmax(self,x):
        return np.exp(x)/np.sum(np.exp(x),axis=0)

    def tanh(self,x):
        self.a=x
        return (np.exp(self.a)-np.exp(-self.a))/(np.exp(self.a)+np.exp(-self.a))
        

    def __init__(self,b,c,W,U,V,X,h1,h2):
        self.b = b
        self.c = c
        self.W = W
        self.U = U
        self.V = V
        self.x = X
        self.h1 = h1
        self.h2 = h2
        self.h1_list=[]
        self.h2_list=[]
        self.h1_list.append(h1)
        self.h2_list.append(h2)
    def get_y_loss(self):


        for t in range(0,3):

            a1=self.b[0]+self.W[0][0]*self.h1 + self.W[0][1]*self.h2 + self.U[0][0]*self.x[t][0]+self.U[0][1]*self.x[t][1]
 #           print('test',self.W[0][0],self.W[0][1])


            a2=self.b[1]+self.W[1][0]*self.h1 + self.W[1][1]*self.h2 + self.U[1][0]*self.x[t][0]+ self.U[1][1]*self.x[t][1]
  #          print('a2',a2)
            self.h1=rnn.tanh(self,a1)
            self.h2=rnn.tanh(self,a2)

            self.h1_list.append(self.h1)
            self.h2_list.append(self.h2)

            o1=self.c[0] + self.V[0][0]*self.h1 + self.V[0][1]*self.h2
            o2=self.c[1] + self.V[1][0]*self.h1 +self.V[1][1]*self.h2
            y1=np.exp(o1)/(np.exp(o1)+np.exp(o2))
            y2=np.exp(o2)/(np.exp(o1)+np.exp(o2))
            loss = np.square(y1-0.5) -np.log(y2)
             
           # print(y1)
           # print(y2)
           # print(loss)
            
        return y1, y2,loss,self.h1_list,self.h2_list


b=np.array([[-1],[1]])
c=np.array([[0.5],[-0.5]])
W=np.array([[1,-1],[0,2]])
U=np.array([[-1,0],[1,-2]])
V=np.array([[-2,1],[-1,0]])

x1=np.array([[1],[0]])
x2=np.array([[0.50],[0.25]])
x3=np.array([[0],[1]])
x=[]
x.append(x1)
x.append(x2)
x.append(x3)
#print(x[1][1])
h1=np.zeros(1)
h2=np.zeros(1)

#using central difference method to get dl/db
b1=np.array([[-0.9999],[1]])
central_get_der =rnn(b1,c,W,U,V,x,h1,h2)
loss1=central_get_der.get_y_loss()[2]
print('loss_b1_1',loss1)

b2=np.array([[-1.0001],[1]])
central_get_der2=rnn(b2,c,W,U,V,x,h1,h2)
loss2=central_get_der2.get_y_loss()[2]
print('loss_b1_2',loss2)
diff_loss = (loss1-loss2)/0.0002
print("central_method_loss:",diff_loss)


#using central difference method to get dl/db2
b1=np.array([[-1],[1.0001]])
central_get_der =rnn(b1,c,W,U,V,x,h1,h2)
loss1=central_get_der.get_y_loss()[2]
print('loss_b2_1',loss1)

b2=np.array([[-1],[0.9999]])
central_get_der2=rnn(b2,c,W,U,V,x,h1,h2)
loss2=central_get_der2.get_y_loss()[2]
print('loss_b2_2',loss2)
diff_loss = (loss1-loss2)/0.0002
print("central_method_loss:",diff_loss)



test=rnn(b,c,W,U,V,x,h1,h2)
y1,y2,loss,h1_list,h2_list=test.get_y_loss()
#print('h1_list',h1_list)
#print('h2_list',h2_list)
print('y1',y1)
print('y2',y2)
print('old loss',loss)

unfolding_loss_b1=test.derivative(b,c,W,U,V,y1,y2,h1_list,h2_list)#by unfolding network
unfolding_loss_b2=test.derivative_b2(b,c,W,U,V,y1,y2,h1_list,h2_list)

print('unfolding_loss_b1:', unfolding_loss_b1)
print('unfolding_loss_b2:', unfolding_loss_b2)

new_b1 = b[0] - 0.002 * unfolding_loss_b1
new_b2 = b[1] - 0.002 * unfolding_loss_b2

print('new_b1,  new_b2', new_b1, new_b2)


new_b=np.array([[new_b1],[new_b2]])

new_test=rnn(new_b,c,W,U,V,x,h1,h2)
y1,y2,loss,h1_list,h2_list=new_test.get_y_loss()


print('new_loss',loss)

