#!/usr/bin/env python
# coding: utf-8

# In[1]:


# NumPy for numerical computing
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import reverse_cuthill_mckee
from fn import *


# In[2]:


def myFE2dbvp(k, f, g, h, maxh, nodes, triangles, edges, bdyNode, bdyEdge, curveEdge, bdyFn, bdyFnder, shapeFn):
    hMax = 100
    while hMax > maxh:
        x01 = nodes[edges[0,0]-1][0]
        x02 = nodes[edges[0,1]-1][0]
        y01 = nodes[edges[0,0]-1][1]
        y02 = nodes[edges[0,1]-1][1]
        hMax = np.sqrt((x02-x01)**2 + (y02-y01)**2)
        for i in range(1,len(edges)):
            xi1 = nodes[edges[i,0]-1][0]
            xi2 = nodes[edges[i,1]-1][0]
            yi1 = nodes[edges[i,0]-1][1]
            yi2 = nodes[edges[i,1]-1][1]
            h_edgei = np.sqrt((xi2-xi1)**2 + (yi2-yi1)**2)
            if h_edgei > hMax:
                hMax = h_edgei
        if hMax > maxh:
            nodes, triangles, edges, bdyNode, bdyEdge, curveEdge = refine(nodes,triangles,edges,
                                                                          bdyNode,bdyEdge,curveEdge,bdyFn,bdyFnder)
    midNodes_val = None
    triangleMidPts_val = None
    edges = np.sort(edges, axis=1)
    
    if shapeFn == 2:
        midNodes_val = midNodes(edges, nodes)
        triangleMidPts_val = triangleMidPts(nodes, triangles, edges)
        
    bdyNodes_val, bdyEdge_val = bdyNodeEdge(nodes, edges, triangles)
    Gamma1Nodes_val = Gamma1Nodes(nodes)
    Gamma2Edges_val = Gamma2Edges(nodes, edges, bdyEdge_val)
    Gamma1MidNodes_val = Gamma1MidNodes(nodes, edges, bdyEdge_val)
    indVec_val = indVec(Gamma1Nodes_val, Gamma1MidNodes_val, shapeFn)
    noOfIntegPt = 7 #default
    noOfIntegPt1d = 5 #default
    K = stiffK2d(k, triangles, nodes, triangleMidPts_val, indVec_val, shapeFn, noOfIntegPt)
    F = loadF2d(k,f,g,h,triangles,nodes,edges,triangleMidPts_val,midNodes_val,indVec_val,Gamma1Nodes_val, 
                    Gamma1MidNodes_val,Gamma2Edges_val,shapeFn,noOfIntegPt,noOfIntegPt1d)
    perm, perm_inv = rcm(K)
    Kbar = K[perm, :][:, perm]
    Fbar = F[perm]
    wbar = spsolve(Kbar, Fbar) 
    w = np.zeros(K.shape[0])
    w[perm] = wbar
    uh = approxSol2d(w, g, indVec_val, nodes, triangles, midNodes_val, triangleMidPts_val, shapeFn)
    graduh = approxSolGrad2d(w, g, indVec_val, nodes, triangles, midNodes_val, triangleMidPts_val, shapeFn)
    return uh, graduh, hMax, nodes, triangles, w


