#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import sin, cos, log10, arctan
import numpy as np
import matplotlib.pyplot as plt
import math

class ABG_Chan:
    alpha=3.53
    betha=22.4
    gamma=2.13
    LOS_exp=70
    shadowing=10
    Coherence_time=1
    def path_loss_dB (self, node1,node2):   #path loss does include antenna gain
        #this is the function that calculates the path loss between two points 
        #P1=(X1,Y1)  P2=(X2,Y2)
        #N1 is the number of antenans at node 1 and N2 is at node 2
        #We assume that they have N1 and N2 beams exactly
        #there are two sync functions that we should multiply


        PI=np.pi
        P1=node1.coords_m
        P2=node2.coords_m
        BB1=node1.beam_book_size
        BB2=node2.beam_book_size
        N1=node1.Num_antennas
        N2=node2.Num_antennas
        index1=node1.current_beam_ID
        index2=node2.current_beam_ID

        X1,Y1=P1[0],P1[1]
        X2,Y2=P2[0],P2[1]
        d_m=pow(pow(X1-X2,2)+ pow(Y1-Y2,2),0.5)
        p_LOS= np.exp(-d_m/self.LOS_exp)
        #print(p_LOS)
        PL=p_LOS*(62+20*np.log10(d_m))+               (1-p_LOS)* (10*self.alpha*np.log10(d_m)+                           self.betha+10*np.log10(28)*self.gamma+self.shadowing)



        eps=1e-8   #only added for numerical stability if the denum goes to zeros
        Phis1=np.linspace(0,np.pi,BB1) #assumption is that
        Phis2=np.linspace(0,np.pi,BB2)
        Phi_opt=np.arctan((Y2-Y1)/(X2-X1+eps)) # This only true if pure LOS
        #np.random.seed(seed)
        mu=0
        sigma=np.pi/4*(1-p_LOS)/self.Coherence_time
        
        Phi_opt+= np.random.normal(mu, sigma, 1)
        #print(Phi_opt)
        #print(Phi_opt)
        si1=PI*sin(Phis1[index1]-Phi_opt)
        si2=PI*sin(Phis2[index2]-Phi_opt-PI)
        Filter1=(sin(N1*si1/2)+eps) /(N1*sin(si1/2)+eps)
        #print(Filter1)
        Filter2=(sin(N2*si2/2)+eps) /(N2*sin(si2/2)+eps)
        #print(Filter2)

        total_PL_dB=PL-10*log10(abs(Filter1*Filter2*N1*N2))

        return(total_PL_dB[0])

def add_dBm(p1_dBm,p2_dBm):
    
    return(10*np.log10(pow(10,0.1*p1_dBm)+pow(10,0.1*p2_dBm)))


class wireless_node:
    
    def __init__(self):
        self.node_type=None
        self.coords_m=(0,0)
        self.beam_book_size=1
        self.Num_antennas=1
        self.NF_dB=0
        self.Rx_sig_power_dBm_per_Hz=-174
        self.Tx_TRP_dBm_per_Hz=-35
        self.Tx_noise_TRP_dBm_per_Hz=-174
        self.current_beam_ID=0
        self.ID=0
        self.SNR_dB=0



class Relay_net:
    
    def __init__(self, coherence_time, mobility_factor):

        self.Nodes=[]
        self.Relays=[]
        self.Weights_dB={} #   Weights[(nodei,nodej)]=w
        self.pairing_Donor={}
        self.Max_FW_gain={}
        self.channel=ABG_Chan()
        self.channel.Coherence_time=coherence_time
        self.channel.UE_mobility=mobility_factor
        
    def insert_node(self,N):
        assert N.node_type in {'UE' ,'gNB'}
        N.ID=len(self.Nodes)
        self.Nodes.append(N)
    
    def insert_repeater_node (self, Relay, Donor, gain_dB, max_gain_dB):
        assert Relay.node_type=='Relay'
        assert Donor.node_type=='Donor'
        Relay.ID=len(self.Nodes)
        self.Nodes.append(Relay)
        Donor.ID=len(self.Nodes)
        self.Nodes.append(Donor)
        self.pairing_Donor[Relay]=(Donor, gain_dB)
        self.Max_FW_gain[(Donor,Relay)]=max_gain_dB
        self.Relays.append(Relay)
        
        
    
    def update_graph_weights(self):
        #claculate the weights of the propagation graph
        
        assert self.Nodes[0].node_type =='gNB'
        for node1 in self.Nodes:
            for node2 in self.Nodes:
                
                if node1.node_type in {'gNB', 'Relay'} and node2.node_type in {'UE', 'Donor'}:
                        if (node1 not in self.pairing_Donor):
                               self.Weights_dB[(node1, node2)]=self.channel.path_loss_dB (node1, node2)
                        elif self.pairing_Donor[node1][0] != node2:
                                self.Weights_dB[(node1, node2)]=self.channel.path_loss_dB (node1, node2)
                 
        for relay in self.Nodes :
            if relay.node_type=='Relay':
                donor=self.pairing_Donor[relay][0]
                self.Weights_dB[(donor,relay)]=-self.pairing_Donor[relay][1]
   

    def calculate_SNR (self):   
        #calculates the SNR of each node given the propagation weights and the gains
        
        
        self.update_graph_weights()
        
        num_nodes=len(self.Nodes)
        W=np.zeros((num_nodes,num_nodes))
        for i in range(num_nodes):
            for j  in range(num_nodes):
                if (self.Nodes[i], self.Nodes[j]) in self.Weights_dB:
                    W[i,j]=pow(10, -0.1*self.Weights_dB[(self.Nodes[i], self.Nodes[j])])
        WT=W.transpose()
        WT[0,0]=1
        for i in range(1, num_nodes):
            WT[i,i]=-1
        
        #print(WT)
        W_inv=np.linalg.inv(WT)
       
        Signals=10*np.log10(W_inv[:,0])+self.Nodes[0].Tx_TRP_dBm_per_Hz
  
        for i in range(len(self.Nodes)):
            if self.Nodes[i].node_type=='Relay':
                    
                    self.Nodes[i].Tx_noise_TRP_dBm_per_Hz=-174+self.pairing_Donor[self.Nodes[i]][1]
      
    
        for i in range(len(self.Nodes)):
           
            self.Nodes[i].Rx_sig_power_dBm_per_Hz=Signals[i]
            self.Nodes[i].Rx_noise_power_dBm_per_Hz=-174+self.Nodes[i].NF_dB
            if self.Nodes[i].node_type=='UE':
                
                for j in range(len(self.Nodes)):
                    if self.Nodes[j].node_type=='Relay':
                        
                        
                        self.Nodes[i].Rx_noise_power_dBm_per_Hz=                        add_dBm(self.Nodes[i].Rx_noise_power_dBm_per_Hz,
                               self.Nodes[j].Tx_noise_TRP_dBm_per_Hz-self.Weights_dB[(self.Nodes[j],self.Nodes[i])])
            
            
            self.Nodes[i].SNR_dB=self.Nodes[i].Rx_sig_power_dBm_per_Hz-                                self.Nodes[i].Rx_noise_power_dBm_per_Hz
            
            
            
        for i in range(len(self.Nodes)):    
             if self.Nodes[i].node_type=='Relay':
                    d=self.pairing_Donor[self.Nodes[i]][0]
                    self.Nodes[i].SNR_dB=d.SNR_dB
 
                    
    def calculate_reward(self):
        self.update_graph_weights()
        self.calculate_SNR()
        R=np.mean([node.SNR_dB for node in self.Nodes if node.node_type=='UE' ])
        return (R)
    
    def update_state(self):
        for node in self.Nodes:
            if node.node_type=='UE':
                x,y= node.coords_m[0],  node.coords_m[1]
                node.coords_m= (x+(np.random.rand()-0.5) *self.channel.UE_mobility , 
                                 y+(np.random.rand()-0.5)* self.channel.UE_mobility)
        self.calculate_SNR ()
        #State includes the current location of all the UEs and their SNR and
        #the SNR and beam_ID of all the Relay nodes and their gains
        
        UE_Coords=[]
        UE_SNRs=[]
        D_SNRs=[]
        D_Beams=[]
        R_Beams=[]
        Gains=[]
        State={}
        for node in self.Nodes:
            
             if node.node_type=='UE':
                    UE_Coords.append(node.coords_m)
                    UE_SNRs.append(node.SNR_dB)
            
        for node in self.Nodes:
            if node.node_type=='Donor':
                D_SNRs.append(node.SNR_dB)
                D_Beams.append(node.current_beam_ID)
            elif node.node_type =='Relay':
                R_Beams.append(node.current_beam_ID)
                Gains.append(self.pairing_Donor[node][1])
        
        State={'UE_SNRs': UE_SNRs, 'UE_Coords': UE_Coords,
           'D_SNRs':D_SNRs, 'D_Beams': D_Beams, 
               'R_Beams': R_Beams, 'Gains': Gains}
        Reward=self.calculate_reward()
        
        return(State,Reward)
        
    def apply_action(self, A):
        
        assert (len(self.Relays)==len(A))
        
        for i in  range(len(self.Relays)):
            node=self.Relays[i]
            action=A[i]
          
            assert action in range(1,8)
            donor=self.pairing_Donor[node][0]

            if action==1:
                node.current_beam_ID+=1
                node.current_beam_ID=node.current_beam_ID%node.beam_book_size
            elif action==2:
                node.current_beam_ID-=1
                node.current_beam_ID=node.current_beam_ID%node.beam_book_size

            elif action==3:
                donor.current_beam_ID+=1
                donor.current_beam_ID=donor.current_beam_ID%donor.beam_book_size
                
                
            elif action==4:
                donor.current_beam_ID-=1
                donor.current_beam_ID=donor.current_beam_ID%donor.beam_book_size
                
            elif action==5:
                self.pairing_Donor[node]=(self.pairing_Donor[node][0],
                                          min(self.pairing_Donor[node][1]+1, self.Max_FW_gain[(donor,node)]))
            elif action==6:
                self.pairing_Donor[node]=(self.pairing_Donor[node][0],
                                          max(self.pairing_Donor[node][1]-1,0))

                
def create_example_env():

    gNB1_1=wireless_node()
    gNB1_1.node_type='gNB'
    gNB1_1.coords_m=(0,0)
    gNB1_1.beam_book_size=1
    gNB1_1.Num_antennas=1
    gNB1_1.Tx_TRP_dBm_per_Hz=-35




    Relay1_1=wireless_node()
    Relay1_1.node_type='Relay'
    Relay1_1.coords_m=(50,-50)
    Relay1_1.beam_book_size=32
    Relay1_1.Num_antennas=16
    Relay1_1.Tx_TRP_dBm_per_Hz=-1000
    Donor1_1=wireless_node()
    Donor1_1.node_type='Donor'
    Donor1_1.coords_m=(50,-50)
    Donor1_1.beam_book_size=64
    Donor1_1.Num_antennas=32
    Donor1_1.NF_dB=5
    Donor1_1.Tx_TRP_dBm_per_Hz=-1000


    Relay2_1=wireless_node()
    Relay2_1.node_type='Relay'
    Relay2_1.coords_m=(50,50)
    Relay2_1.beam_book_size=32
    Relay2_1.Num_antennas=16
    Relay2_1.Tx_TRP_dBm_per_Hz=-1000
    Donor2_1=wireless_node()
    Donor2_1.node_type='Donor'
    Donor2_1.coords_m=(50,50)
    Donor2_1.beam_book_size=64
    Donor2_1.Num_antennas=32
    Donor2_1.NF_dB=5
    Donor2_1.Tx_TRP_dBm_per_Hz=-1000





    UE1_1=wireless_node()
    UE1_1.node_type='UE'
    UE1_1.coords_m=(-100,80)
    UE1_1.beam_book_size=1
    UE1_1.Num_antennas=1
    UE1_1.Tx_TRP_dBm_per_Hz=-1000
    UE1_1.NF_dB=5

    UE2_1=wireless_node()
    UE2_1.node_type='UE'
    UE2_1.coords_m=(0,-45)
    UE2_1.beam_book_size=1
    UE2_1.Num_antennas=1
    UE2_1.Tx_TRP_dBm_per_Hz=-1000
    UE2_1.NF_dB=5

    UE3_1=wireless_node()
    UE3_1.node_type='UE'
    UE3_1.coords_m=(30,90)
    UE3_1.beam_book_size=1
    UE3_1.Num_antennas=1
    UE3_1.Tx_TRP_dBm_per_Hz=-1000
    UE3_1.NF_dB=5

    UE4_1=wireless_node()
    UE4_1.node_type='UE'
    UE4_1.coords_m=(-60,-35)
    UE4_1.beam_book_size=1
    UE4_1.Num_antennas=1
    UE4_1.Tx_TRP_dBm_per_Hz=-1000
    UE4_1.NF_dB=5


    relay_net1=Relay_net(coherence_time=1e9, mobility_factor=0)
    relay_net1.insert_node(gNB1_1)
    relay_net1.insert_node(UE1_1)
    relay_net1.insert_node(UE2_1)
    relay_net1.insert_node(UE3_1)
    relay_net1.insert_node(UE4_1)
    relay_net1.insert_repeater_node (Relay1_1, Donor1_1, 100,110)
    relay_net1.insert_repeater_node (Relay2_1, Donor2_1, 100,110)
    ##############################################################
    gNB1_2=wireless_node()
    gNB1_2.node_type='gNB'
    gNB1_2.coords_m=(0,0)
    gNB1_2.beam_book_size=1
    gNB1_2.Num_antennas=1
    gNB1_2.Tx_TRP_dBm_per_Hz=-35




    Relay1_2=wireless_node()
    Relay1_2.node_type='Relay'
    Relay1_2.coords_m=(50,-50)
    Relay1_2.beam_book_size=32
    Relay1_2.Num_antennas=16
    Relay1_2.Tx_TRP_dBm_per_Hz=-1000
    Donor1_2=wireless_node()
    Donor1_2.node_type='Donor'
    Donor1_2.coords_m=(50,-50)
    Donor1_2.beam_book_size=64
    Donor1_2.Num_antennas=32
    Donor1_2.NF_dB=5
    Donor1_2.Tx_TRP_dBm_per_Hz=-1000


    Relay2_2=wireless_node()
    Relay2_2.node_type='Relay'
    Relay2_2.coords_m=(50,50)
    Relay2_2.beam_book_size=32
    Relay2_2.Num_antennas=16
    Relay2_2.Tx_TRP_dBm_per_Hz=-1000
    Donor2_2=wireless_node()
    Donor2_2.node_type='Donor'
    Donor2_2.coords_m=(50,50)
    Donor2_2.beam_book_size=64
    Donor2_2.Num_antennas=32
    Donor2_2.NF_dB=5
    Donor2_2.Tx_TRP_dBm_per_Hz=-1000





    UE1_2=wireless_node()
    UE1_2.node_type='UE'
    UE1_2.coords_m=(-100,80)
    UE1_2.beam_book_size=1
    UE1_2.Num_antennas=1
    UE1_2.Tx_TRP_dBm_per_Hz=-1000
    UE1_2.NF_dB=5

    UE2_2=wireless_node()
    UE2_2.node_type='UE'
    UE2_2.coords_m=(0,-45)
    UE2_2.beam_book_size=1
    UE2_2.Num_antennas=1
    UE2_2.Tx_TRP_dBm_per_Hz=-1000
    UE2_2.NF_dB=5

    UE3_2=wireless_node()
    UE3_2.node_type='UE'
    UE3_2.coords_m=(30,90)
    UE3_2.beam_book_size=1
    UE3_2.Num_antennas=1
    UE3_2.Tx_TRP_dBm_per_Hz=-1000
    UE3_2.NF_dB=5

    UE4_2=wireless_node()
    UE4_2.node_type='UE'
    UE4_2.coords_m=(-60,-35)
    UE4_2.beam_book_size=1
    UE4_2.Num_antennas=1
    UE4_2.Tx_TRP_dBm_per_Hz=-1000
    UE4_2.NF_dB=5


    relay_net2=Relay_net(coherence_time=1e5, mobility_factor=0.5)
    relay_net2.insert_node(gNB1_2)
    relay_net2.insert_node(UE1_2)
    relay_net2.insert_node(UE2_2)
    relay_net2.insert_node(UE3_2)
    relay_net2.insert_node(UE4_2)
    relay_net2.insert_repeater_node (Relay1_2, Donor1_2, 100,110)
    relay_net2.insert_repeater_node (Relay2_2, Donor2_2, 100,110)
    
    return (relay_net1, relay_net2)

