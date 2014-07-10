# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 14:06:55 2014

@author: ankur
"""

import csv
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pylab as pl,random
from math import exp


def draw_figure(fignum,data,pos):
    
    fig = pl.figure(fignum)
    pl.clf()
    ax = fig.add_subplot(111, projection='3d')
    #ax = Axes3D(fig,elev=48, azim=134)
    x,y,z = zip(*data)
    ax.scatter(x,y,z,c='#ffffff',zorder=0,s=5)
    #pass the data here from the graph
    x1,y1,z1= zip(*pos)
    ax.plot(x1,y1,z1,zorder=1,lw=1)    
    ax.scatter(x1,y1,z1,c="r",zorder=1,alpha=1,s=20)
    
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('X coordinates')
    ax.set_ylabel('Y coordinates')
    ax.set_zlabel('Z coordinates')
    ax.view_init(40,  50) 
    ax.set_title('Self Organizing Map')
    
    pl.savefig(str(fignum)+'.png')
    pl.close(fignum)
    fignum = fignum + 1

def read_csv_get_coordinate(filename):
        """reads a csv file and returns an enumerable object"""
        data =[]
        with open(filename, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                    row[0]=float(row[0])
                    row[1] = float(row[1])
                    row[2] = float(row[2])
                    data.append([row[0],row[1],row[2]])
        return data
        
class CSom():
    
    def __init__(self,data,num_of_nodes=15):
        self.a = None
        self.data = data
        self.codebook_vectors = []
        self.graph = nx.Graph()
        self.count_of_neurons = 0
        
        self.pos = None
        self.tmax = None
        self.eps_e =None #eta
        self.eps_n = None #neeta
        self.dist_mat = None
        
        self.imagecount = 0
        
        #initialize codebook vectors from here
        temp_list = np.random.random_integers(0, len(data),num_of_nodes)
        
        temp_list = sorted(temp_list)
        
        #so we have  nodes in our labelled graph now        
        for i in temp_list:        
            self.graph.add_node(self.count_of_neurons, pos=(self.data[i]))
            self.count_of_neurons+=1
            
        for i in xrange(0,len(temp_list)):
            if(i == len(temp_list)-1):
                self.graph.add_edge(0,i)
            else:
                self.graph.add_edge(i,i+1)
        print self.graph.nodes(), self.graph.edges()
        self.dist_mat = self.get_distance_matrix()
        
    def get_distance_matrix(self):
        
        vertices = self.graph.nodes()
        distance_matrix = []
        #path length of two vertices
        size_of_mat = len(vertices)
        for i in xrange(0,size_of_mat):
            distance_matrix.append([])
            for j in xrange(0,size_of_mat):
                path_len = len(nx.shortest_path(self.graph,i,j))-1
                #print i, j,path_len
                distance_matrix[i].append(path_len)
        return distance_matrix    

    def euclidean_distance(self, a,b):
        
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)**0.5
        
    def find_closest_codebookvector(self,x):
        """where this curnode is actually the x,y index of the data we want to analyze"""
        min_dist = float("inf")
        self.pos = nx.get_node_attributes(self.graph,'pos')
        #print self.pos        
        for node,coordinates in self.pos.iteritems():
            dist = self.euclidean_distance(x,coordinates)
            if  dist < min_dist:
                min_dist = dist
                min_node = node
                coord_of_bmu= coordinates
                
        return min_node, coord_of_bmu    
    
    def get_new_pos(self,sampx,vector,Dij):
                   
        mulfactor1= [(sampx[0] - vector[0]),(sampx[1]-vector[1]),(sampx[2]-vector[2])]
        mul_factor2 = exp((-float(Dij)/float(2*self.eps_n))) 
        
        net_move = [mulfactor1[0]*mul_factor2*self.eps_e, mulfactor1[1]*mul_factor2*self.eps_e,mulfactor1[2]*mul_factor2*self.eps_e]
        
        new_position = [vector[0]+net_move[0], vector[1]+ net_move[1], vector[2]+net_move[2]]
        return new_position
        
    def update_vectors(self,sample_x):
         
         #determine the winner neuron 
         winnernode, win_node_coord = self.find_closest_codebookvector(sample_x)
         #print "winner node is", winnernode
         self.pos = nx.get_node_attributes(self.graph,'pos')
         
         #update all the codebook vectors now
         for node in self.graph.nodes():
             pos_of_node= self.pos[node]
             newpos = self.get_new_pos(sample_x,pos_of_node,self.dist_mat[winnernode][node])
             
             self.graph.add_node(node, pos=newpos)
            
         
         
        
    def train(self,max_iterations=10000):
        self.tmax = max_iterations
        fignum=0
        stepsize =50
        
       
        self.draw_updated_fig(fignum)
        for i in xrange(0,max_iterations+1):
            print "Iterating..{0:d}".format(i)
            div_factor = (float(i)/float(self.tmax))
            self.eps_e = (1- div_factor)
            self.eps_n = exp(-div_factor)
                       
            random.shuffle(self.data)
            for x in self.data:
                self.update_vectors(x)
           
        
           #start plotting only after it has learned a presentable amount                     
            check = i%stepsize
            if check==0:
                fignum+=1
                self.draw_updated_fig(fignum)
           
            
    def draw_updated_fig(self,fignum):
        dictn= nx.get_node_attributes(self.graph,'pos')
               
        pos =[] 
        for node,coordinates in dictn.iteritems():
            #print "nodes, coordinates",node, coordinates
            pos.append(coordinates)
            #connect the last to first node again
        pos.append(dictn[0])
        draw_figure(fignum,data,pos)    


            
if __name__ == '__main__':

    filename = "path2.csv"
    data = read_csv_get_coordinate(filename) 
    obj = CSom(data,num_of_nodes = 40)
    if obj is not None:
        obj.train(10000)
    f= open('output_si_values.txt','w')
    positions = nx.get_node_attributes(obj.graph,'pos')
    for i,j in positions.iteritems():
        f.write(str(j)+"\n")
    f.close()
 
    
    