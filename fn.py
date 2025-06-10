#!/usr/bin/env python
# coding: utf-8

# In[1]:


# NumPy for numerical computing
import numpy as np
# SciPy sparse matrices
import scipy.sparse as sp
# SciPy sparse matrices
import scipy.optimize as spo


# In[2]:

# Perform LU decomposition optimized for band matrices
def bandLU(K, p, q):
    n = np.shape(K)[0]
    for m in range(0, n-1):
        for i in range(m+1, min(m+p+1, n)):
            K[i,m] = K[i,m]/K[m,m]
        for j in range(m+1, min(m+q+1, n)):
            for i in range(m+1, min(m+p+1, n)):
                K[i,j] = K[i,j] - K[i,m]*K[m,j]
    return K

# Forward substitution for banded lower triangular matrix
def bandforward(L, f, p):
    n = np.shape(L)[0]
    for i in range(1, n):
        for j in range(max(0, i-p), i):
            f[i] = f[i]-L[i,j]*f[j]
    return f

# Backward substitution for banded upper triangular matrix
def bandbackward(U, f, q):
    n = np.shape(U)[0]
    for i in range(n-1, -1, -1):
        for j in range(i+1, min(i+q+1, n)):
            f[i] = f[i] - U[i,j]*f[j]
        f[i] = f[i]/U[i,i]
    return f

# Gauss quadrature integration (1D)
def gaussQuadStd1d(g, noOfIntegPt):
    if noOfIntegPt == 2:
        y = g(-1/((3)**0.5)) + g(1/((3)**0.5))
    if noOfIntegPt == 3:
        y = 5/9 * g(-(3/5)**0.5) + 8/9 * g(0) + 5/9 * g((3/5)**0.5)
    if noOfIntegPt == 4:
        x1 = - ((3/7) + (2/7)*(6/5)**0.5)**0.5
        x2 = - ((3/7) - (2/7)*(6/5)**0.5)**0.5
        x3 = - x2
        x4 = - x1
        w1 = w4 = (18 - (30)**0.5) / 36
        w2 = w3 = (18 + (30)**0.5) / 36
        y = w1 * g(x1) + w2 * g(x2) + w3 * g(x3) + w4 * g(x4)
    if noOfIntegPt == 5:
        w1 = w5 = (322 - 13 * (70)**0.5) / 900
        w2 = w4 = (322 + 13 * (70)**0.5) / 900
        w3 = 128/225
        x1 = - (5 + 2*(10/7)**0.5)**0.5 / 3
        x2 = - (5 - 2*(10/7)**0.5)**0.5 / 3
        x4 = - x2
        x5 = - x1
        y = w1 * g(x1) + w2 * g(x2) + w3 * g(0) + w4 * g(x4) + w5 * g(x5)
    return y

# Gauss quadrature integration (1D)
def gaussQuad1d(fn, lowerLimit, upperLimit, noOfIntegPt):
    def g(xi):
        x = 0.5 * (upperLimit - lowerLimit) * (xi + 1) + lowerLimit
        return fn(x) * (upperLimit - lowerLimit) / 2.0
    y = gaussQuadStd1d(g, noOfIntegPt)
    return y

# Find midpoint on curved boundary using arc-length
def midPtCurve(f, fder, x1, x2):
    integrand = lambda x: (1 + fder(x)**2)**0.5
    L = gaussQuad1d(integrand, x1, x2, 5)
    def m(xm):
        integral = gaussQuad1d(integrand, x1, xm, 5)
        return integral - L / 2
    xm = spo.brentq(m, x1, x2)
    ym = f(xm)
    return (xm, ym)


# In[3]:


# Refine triangular mesh by splitting triangles
def refine(nodes,triangles,edges,bdyNode,bdyEdge,curveEdge,bdyFn,bdyFnder):
    '''
    Refine a domain

    nodes,triangles,edges,bdyNode,bdyEdge,curveEdge =refine(nodes,triangles,edges,bdyNode,bdyEdge,curveEdge,bdyFn,bdyFnder)

    this program refines a triangulation using standard refinement 
    input and output:
    nodes - 2-column array that stores the (x,y)-coordinates of nodes
    trangles - 3-column array that stores the nodes in each triangle
    edges - 2-column array that stores the end nodes of edges
    bdyNode - 1D array that indicates if a node is a boundary node  
    bdyEdge - 1D array that indicates if an edge is a boundary edge 
    curveEdge - 1D array that indicates if an edge is an approximation of a curve 
    bdyFn - a function represents the function that describe the boundary
    bdyFnder - the derivative of the boundary function
    
    '''
    
    noTri = np.shape(triangles)[0] # no of triangles
    noNode = np.shape(nodes)[0] # no of nodes
    parentNodes = np.array([]).reshape((0,2))   # track where the midpoint is from

    # go through triangles to get midpoints
    for k in range(0,noTri):
        node1 = triangles[k,0]
        node2 = triangles[k,1]
        node3 = triangles[k,2]
    
        # check the first edge
        inEdges = (np.sum((edges == node1)+(edges == node2),axis=1)==2) # find where the first edge is in the edge list
        if (np.sum(inEdges)): # if found
            edgeNo = np.where(inEdges)[0] # which edge it is

            # find midpoint
            if (curveEdge[edgeNo[0]] == 1):  # if it is a curve edge
                x1m,y1m = midPtCurve(bdyFn,bdyFnder,nodes[node1-1,0],nodes[node2-1,0])
            else:
                x1m = 0.5*(nodes[node1-1,0]+nodes[node2-1,0])
                y1m = 0.5*(nodes[node1-1,1]+nodes[node2-1,1])
            # endif curveEdge
            
            nodes,edges,bdyNode,bdyEdge,curveEdge = updateTri(x1m,y1m,node1,node2,edgeNo,nodes,edges,bdyNode,bdyEdge,curveEdge)
            numNode = len(nodes[:,0])
            nodeNo4MidPt1 = numNode # record the node number of the 1st midpoint
            parentNodes = np.vstack((parentNodes,[node1,node2])) # record the parents that produce this midpoints
                    
        else: # not in existing edges
            # find the node number for the 1st midpoint
            nodeNo4MidPt1 = noNode + np.where(np.sum((parentNodes==node1)+(parentNodes==node2),axis=1)==2)[0][0]+1
        # endif inEdges
            
    
        # check the second edge
        inEdges = (np.sum((edges==node2)+(edges==node3),axis=1)==2) # find where the second edge is in the edge list
        if (np.sum(inEdges)): # if found
            edgeNo = np.where(inEdges)[0] # which edge it is
        
            # find midpoint
            if (curveEdge[edgeNo[0]] == 1): # if it is a curve edge
                x2m,y2m = midPtCurve(bdyFn,bdyFnder,nodes[node2-1,0],nodes[node3-1,0])
            else:
                x2m = 0.5*(nodes[node2-1,0]+nodes[node3-1,0])
                y2m = 0.5*(nodes[node2-1,1]+nodes[node3-1,1])
            # endif curveEdge
            
            nodes,edges,bdyNode,bdyEdge,curveEdge = updateTri(x2m,y2m,node2,node3,edgeNo,nodes,edges,bdyNode,bdyEdge,curveEdge)
            numNode = len(nodes[:,0])
            nodeNo4MidPt2 = numNode # record the node number of the 2nd midpoint
            parentNodes = np.vstack((parentNodes,[node2, node3])) # record the parents that produce this midpoints
            
        else: # not in existing edges
            # find the node number for the 2nd midpoint
            nodeNo4MidPt2 = noNode + np.where(np.sum((parentNodes==node2)+(parentNodes==node3),axis=1)==2)[0][0]+1
        # endif inEdges
            

        # check the third edge
        inEdges = (np.sum((edges==node1)+(edges==node3),axis=1)==2) # find where the third edge is in the edge list
        if (np.sum(inEdges)): # if found
            edgeNo = np.where(inEdges)[0] # which edge it is
        
            # find midpoint
            if (curveEdge[edgeNo[0]] == 1): # if it is a curve edge
                x3m,y3m = midPtCurve(bdyFn,bdyFnder,nodes[node1-1,0],nodes[node3-1,0])
            else:
                x3m = 0.5*(nodes[node1-1,0]+nodes[node3-1,0])
                y3m = 0.5*(nodes[node1-1,1]+nodes[node3-1,1])
            # endif curveEdge
            
            nodes,edges,bdyNode,bdyEdge,curveEdge = updateTri(x3m,y3m,node1,node3,edgeNo,nodes,edges,bdyNode,bdyEdge,curveEdge)
            numNode = len(nodes[:,0])
            nodeNo4MidPt3 = numNode # record the node number of the 3rd midpoint
            parentNodes = np.vstack((parentNodes,[node1, node3])) # record the parents that produce this midpoints
            
        else: # not in existing edges
            # find the node number for the 3rd midpoint
            nodeNo4MidPt3 = noNode + np.where(np.sum((parentNodes==node1)+(parentNodes==node3),axis=1)==2)[0][0]+1
        # endif inEdges
        
  
        # put the 3 new edges in (cnnecting 3 midpoints)
        edges = np.vstack((edges,[[nodeNo4MidPt1, nodeNo4MidPt2],[nodeNo4MidPt2, nodeNo4MidPt3],[nodeNo4MidPt1, nodeNo4MidPt3]]))
        bdyEdge = np.hstack((bdyEdge,[0,0,0]))
        curveEdge = np.hstack((curveEdge,[0,0,0]))

        # replace this triangle in the triangle list by 4 new triangles
        triangles[k,:] = np.array([node1, nodeNo4MidPt1, nodeNo4MidPt3])
        triangles = np.vstack((triangles,[node2, nodeNo4MidPt2, nodeNo4MidPt1]))
        triangles = np.vstack((triangles,[node3, nodeNo4MidPt3, nodeNo4MidPt2]))
        triangles = np.vstack((triangles,[nodeNo4MidPt1, nodeNo4MidPt2, nodeNo4MidPt3]))
    # endfor all triangles

    return nodes, triangles, edges, bdyNode, bdyEdge, curveEdge


# In[4]:


# Update mesh data structures after inserting midpoint
def updateTri(xm,ym,parentNode1,parentNode2,edgeNo,nodes,edges,bdyNode,bdyEdge,curveEdge):
    '''
    The function that updates nodes, edges, bdyNode, bdyEdge, curveEdge when doing refinement

    [nodes,edges,bdyNode,bdyEdge,curveEdge] = updateTri(xm,ym,parentNode1,parentNode2,edgeNo,nodes,edges,bdyNode,bdyEdge,curveEdge)

    Chung-min Lee Mar 26, 2025
    '''

    # add the midpoint to nodes
    nodes = np.vstack((nodes, [xm, ym]))
    row = len(nodes[:,1])

    # replace the original edge by 2 new edges
    edges[edgeNo,:] = np.array([parentNode1, row])
    edges = np.vstack((edges,[parentNode2, row]))

    # if the parent nodes are on a bdyEdge, then this midpoint is a boundary
    # node and both new edges are boundary edges
    if (bdyEdge[edgeNo] == 1):
        bdyNode = np.hstack((bdyNode,[1]))
        bdyEdge = np.hstack((bdyEdge,[1]))
    else:
        bdyNode = np.hstack((bdyNode,[0]))
        bdyEdge = np.hstack((bdyEdge,[0]))
    # endif boundary edge

    # update curveEdge for two new edges
    if (curveEdge[edgeNo] == 1):
        curveEdge = np.hstack((curveEdge,[1]))
    else:
        curveEdge = np.hstack((curveEdge,[0]))
    # endif curveEdge
    
    return nodes,edges,bdyNode,bdyEdge,curveEdge


# In[5]:


# Evaluate 2D shape functions for triangle
def shapeFn2dTs(i: int,x,y,p: int):
    '''
    this function gives the linear and quadratic shape functions on a
    standard triangle element with vertices (0,0), (0,1) and (1,0)
    [z] = shapeFn2dTs(i,x,y,p)
    input:
        i indicates the ith shape function psi_i, scalar
        x,y are the input variables, in np arrays
        p is the order of shape function
    output:
        z
    '''
    
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # find which points are inside the standard triangle
    ind =  (x>= -1.0e-12) & (y >= -1.0e-12) & (y+x -1 <= 1.0e-12)
   

    # prepare output array
    z = np.zeros(x.shape)
    
    
    if (p == 1): # linear shape function
        match i:
            case 1:
                z[ind] = 1 - x[ind] - y[ind]
            case 2:
                z[ind] = x[ind]
            case 3:
                z[ind] = y[ind]
            case _:
                print(f' wrong input on i')
        # end match
    else: # p==2 quadratic shape function
        match i:
            case 1:
                z[ind] = (1. - x[ind] - y[ind]) * (1. - 2.*x[ind] - 2.*y[ind])
            case 2:
                z[ind] = x[ind] * (2.*x[ind] - 1.)
            case 3:
                z[ind] = y[ind] * (2.*y[ind] - 1.)
            case 4:
                z[ind] = 4. * x[ind] * (1.-x[ind]-y[ind])
            case 5:
                z[ind] = 4. * x[ind] * y[ind]
            case 6:
                z[ind] = 4. * y[ind] * (1. -x[ind]-y[ind])
            case _:
                print(f' wrong input on i')
        # end match
    # end if shape function order
    return z

# Evaluate 2D shape functions for triangle
def shapeFn2d(i: int,x,y,x0,y0,x1,y1,x2,y2,p: int):
    '''
    this function gives the linear and quadratic shape functions on a triangle
    element with vertices (x0,y0), (x1,y1) and (x2,y2)
 
    [z] = shapeFn2d(i,x,y,x0,y0,x1,y1,x2,y2,p)
    input:
        i indicates the ith shape function psi_i, scalar
        x,y are the input variables, in np arrays
        x0, y0, x1, y1, x2, y2 are coordinates of the vertices
        p is the order of shape functions
    output:
        z
    '''

    XI = ((y2-y0)*(x-x0)+(x0-x2)*(y-y0)) / ((x1-x0)*(y2-y0)-(x2-x0)*(y1-y0))
    ETA =((y0-y1)*(x-x0)+(x1-x0)*(y-y0)) / ((x1-x0)*(y2-y0)-(x2-x0)*(y1-y0))


    z = shapeFn2dTs(i,XI,ETA,p)
    return z



# Gradient of shape functions in physical coordinates
def shapeFnGrad2dTs(i:int,xi,eta,p:int):
    '''
    this program gives the derivatives of the linear and quadratic shape functions on the
    standard triangle element with verivces (0,0), (1,0) and (0,1)
 
    [psi_xi,psi_eta] = shapeFnGrad2dTs(i,xi,eta,p)

    input:
        i indicates the ith shape function psi_i
        xi,eta are the input variables, in np arrays
        p is the order of shape function
    output:
        psi_xi, psi_eta
    '''
    
    xi = np.atleast_1d(xi)
    eta = np.atleast_1d(eta)
    
    # find which points are inside the standard triangle
    ind =  (xi>= -1.0e-12) & (eta >= -1.0e-12) & (eta+xi -1 <= 1.0e-12)      
   

    # prepare output array
    psi_xi = np.zeros(xi.shape)
    psi_eta = np.zeros(eta.shape)
 
    # shape function order
    if (p == 1): # linear shape function
        match i: # which shape function
            case 1:
                psi_xi[ind] = -1.
                psi_eta[ind] = -1.
            case 2:
                psi_xi[ind] = 1.
                psi_eta[ind] = 0.
            case 3:
                psi_xi[ind] = 0.
                psi_eta[ind] = 1.
            case _:
                print(f' wrong input on i')
        # end match 
    else: # p== 2 quadratic shape function
        match i: # which shape function
            case 1:
                psi_xi[ind] = -3. + 4.*xi[ind] + 4.*eta[ind]
                psi_eta[ind] = -3. + 4.*xi[ind] + 4.*eta[ind]
            case 2:
                psi_xi[ind] = 4.*xi[ind] - 1.
                psi_eta[ind] = 0.
            case 3:
                psi_xi[ind] = 0.
                psi_eta[ind] = 4.*eta[ind] - 1.
            case 4:
                psi_xi[ind] = 4. - 8.*xi[ind] - 4.*eta[ind]
                psi_eta[ind] = -4.*xi[ind]
            case 5:
                psi_xi[ind] = 4.*eta[ind]
                psi_eta[ind] = 4.*xi[ind]
            case 6:
                psi_xi[ind] = -4.*eta[ind]
                psi_eta[ind] = 4. - 4.*xi[ind] - 8.*eta[ind]
            case _:
                print(f' wrong input on i')
        #end match
    # end shape function order
    
    return psi_xi, psi_eta

# Gradient of shape functions in physical coordinates
def shapeFnGrad2d(i:int,x,y,x0,y0,x1,y1,x2,y2,p:int):
    '''
     this program gives the derivatives of the linear and quadratic shape functions on a 
    triangle element with verivces (x0,y0), (x1,y1) and (x2,y2)
    [gradpsi] = shapeFnGrad2d(i,x,y,x0,y0,x1,y1,x2,y2,p)
 
    input:
        i indicates the ith shape function psi_i
        x,y are the input variables, in np arrays
        x0, y0, x1, y1, x2, y2 are coordinates of vertices 
        p is the order of shape function
    output:
        gradpsi, in 2-row np array

    '''

    # map to standard triangle coordinates
    xi = ((y2-y0) * (x-x0) + (x0-x2) * (y-y0)) / ((x1-x0) * (y2-y0) - (x2-x0) * (y1-y0))
    eta =((y0-y1) * (x-x0) + (x1-x0) * (y-y0)) / ((x1-x0) * (y2-y0) - (x2-x0) * (y1-y0))

    #print(f'{xi},{eta}')

    # find deriavtives on standard triangle
    a,b=shapeFnGrad2dTs(i,xi,eta,p)

    # multiply by Jacobian
    psi_x = (((y2-y0) * a + (y0-y1) * b) / ((x1-x0) * (y2-y0) - (x2-x0) * (y1-y0)))
    psi_y = (((x0-x2) * a + (x1-x0) * b) / ((x1-x0) * (y2-y0) - (x2-x0) * (y1-y0)))
    gradpsi = np.vstack((psi_x, psi_y))

    return gradpsi


# In[6]:


def quad2dTs(g, noOfIntegPt):
    if noOfIntegPt == 4:
        val = -9/32*g(1/3,1/3)+25/96*g(3/5,1/5)+25/96*g(1/5,1/5)+25/96*g(1/5,3/5)
    elif noOfIntegPt == 6:
        w1 = 137/2492
        w4 = 1049/9392
        a = 1280/1567
        b = 287/3134
        c = 575/5319
        d = 2372/5319
        val = w1*g(a,b)+w1*g(b,a)+w1*g(b,b)+w4*g(c,d)+w4*g(d,c)+w4*g(d,d)
    elif noOfIntegPt == 7:
        w1 = 9/80
        w2 = 352/5590
        w5 = 1748/26406
        a = 1/3
        b = 248/311
        c = 496/4897
        d = 248/4153
        e = 496/1055
        val = w1*g(a,a)+w2*g(b,c)+w2*g(c,b)+w2*g(c,c)+w5*g(d,e)+w5*g(e,d)+w5*g(e,e)
    return val

# Quadrature over a triangle in 2D
def quad2dTri(f, x1, y1, x2, y2, x3, y3, noOfIntegPt):
    detA = (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)
    g = lambda xi, eta: f((x2-x1)*xi+(x3-x1)*eta + x1, (y2-y1)*xi+(y3-y1)*eta + y1)
    val = abs(detA)*quad2dTs(g, noOfIntegPt)
    return val


# In[7]:


# Compute stiffness matrix entry for element
def keij2d(k,e,i,j,triangles,nodes,shapeFn,noOfIntegPt):
    z1 = 0
    z2 = 0
    z3 = 0
    
    [x1, y1] = nodes[triangles[e-1,0]-1]
    [x2, y2] = nodes[triangles[e-1,1]-1]
    [x3, y3] = nodes[triangles[e-1,2]-1]
    
    gradpsi = lambda x, y: shapeFnGrad2d(i, x, y, x1, y1, x2, y2, x3, y3, shapeFn)
    gradpsj = lambda x, y: shapeFnGrad2d(j, x, y, x1, y1, x2, y2, x3, y3, shapeFn)
    intgr = lambda x, y: k(x,y)*(gradpsi(x,y)[0]*gradpsj(x,y)[0]+gradpsi(x,y)[1]*gradpsj(x,y)[1])
    z = quad2dTri(intgr, x1, y1, x2, y2, x3, y3, noOfIntegPt)
    return z

# Compute load vector entry for element
def fei2d(k,f,g,h,e,i,triangles,nodes,edges,triangleMidPts,midNodes,Gamma1Nodes,
          Gamma1MidNodes,Gamma2Edges,shapeFn,noOfIntegPt,noOfIntegPt1d):
    z1 = 0
    z2 = 0
    z3 = 0
    nnodes = np.shape(nodes)[0]
    
    [x1, y1] = nodes[triangles[e-1,0]-1]
    [x2, y2] = nodes[triangles[e-1,1]-1]
    [x3, y3] = nodes[triangles[e-1,2]-1]
    
    psi = lambda x, y: shapeFn2d(i, x, y, x1, y1, x2, y2, x3, y3, shapeFn)
    gradpsi = lambda x, y: shapeFnGrad2d(i, x, y, x1, y1, x2, y2, x3, y3, shapeFn)
    
    intgr1 = lambda x, y: f(x,y)*psi(x,y)
    z1 = quad2dTri(intgr1, x1, y1, x2, y2, x3, y3, noOfIntegPt)

    g_nodes = np.copy(Gamma1Nodes).astype(float)
    g_nodes *= [g(*node) for node in nodes]
    if shapeFn ==2:
        g_midnodes = np.copy(Gamma1MidNodes).astype(float)
        g_midnodes *= [g(*node) for node in midNodes]
    for j in [1,2,3]:
        gradpsj = lambda x, y: shapeFnGrad2d(j, x, y, x1, y1, x2, y2, x3, y3, shapeFn)
        intgr2 = lambda x, y: k(x,y)*(gradpsi(x,y)[0]*gradpsj(x,y)[0]+gradpsi(x,y)[1]*gradpsj(x,y)[1])
        z2 += g_nodes[triangles[e-1,j-1]-1]*quad2dTri(intgr2, x1, y1, x2, y2, x3, y3, noOfIntegPt)
    if shapeFn == 2:
        for j in [4,5,6]:
            gradpsj = lambda x, y: shapeFnGrad2d(j, x, y, x1, y1, x2, y2, x3, y3, shapeFn)
            intgr2 = lambda x, y: k(x,y)*(gradpsi(x,y)[0]*gradpsj(x,y)[0]+gradpsi(x,y)[1]*gradpsj(x,y)[1])
            z2 += g_midnodes[triangleMidPts[e-1,j-4]-nnodes-1]*quad2dTri(intgr2, x1, y1, x2, y2, x3, y3, noOfIntegPt)
    
    if Gamma2Edges[np.where((edges == np.sort(triangles[e-1,0:2])).all(axis=1))[0][0]]: #edge (x1,y1) to (x2,y2) in Gamma2
        intgr3 = lambda t: h(x1+t*(x2-x1),y1+t*(y2-y1))*psi(x1+t*(x2-x1),y1+t*(y2-y1))
        z3 += np.sqrt((x2-x1)**2+(y2-y1)**2)*gaussQuad1d(intgr3, 0, 1, noOfIntegPt1d)
        
    if Gamma2Edges[np.where((edges == np.sort(triangles[e-1,1:3])).all(axis=1))[0][0]]: #edge (x2,y2) to (x3,y3) in Gamma2
        intgr3 = lambda t: h(x2+t*(x3-x2),y2+t*(y3-y2))*psi(x2+t*(x3-x2),y2+t*(y3-y2))
        z3 += np.sqrt((x3-x2)**2+(y3-y2)**2)*gaussQuad1d(intgr3, 0, 1, noOfIntegPt1d)
        
    if Gamma2Edges[np.where((edges == np.sort(triangles[e-1,0:3:2])).all(axis=1))[0][0]]: #edge (x1,y1) to (x3,y3) in Gamma2
        intgr3 = lambda t: h(x3+t*(x1-x3),y3+t*(y1-y3))*psi(x3+t*(x1-x3),y3+t*(y1-y3))
        z3 += np.sqrt((x1-x3)**2+(y1-y3)**2)*gaussQuad1d(intgr3, 0, 1, noOfIntegPt1d)
        
    z = z1-z2+z3
    return z

# Assemble global stiffness matrix
def stiffK2d(k, triangles, nodes, triangleMidPts, indVec, shapeFn, noOfIntegPt):
    K = np.zeros((np.count_nonzero(indVec),np.count_nonzero(indVec)))
    if shapeFn == 1:
        tris = triangles
    elif shapeFn ==2:
        tris = np.hstack((triangles, triangleMidPts))
    data = np.array([])
    row = np.array([])
    col = np.array([])
    for e in range(1, np.shape(tris)[0]+1):
        for i in range(1, 3*shapeFn+1):
            for j in range(1, i+1):
                if indVec[tris[e-1,i-1]-1]*indVec[tris[e-1,j-1]-1] != 0:
                    k_val = keij2d(k,e,i,j,triangles,nodes,shapeFn,noOfIntegPt)[0]
                    if abs(k_val) > 1e-12:
                        row = np.append(row,indVec[tris[e-1,i-1]-1]-1)
                        col = np.append(col,indVec[tris[e-1,j-1]-1]-1)
                        data = np.append(data,k_val)
                        if row[-1] != col[-1]: #not on diagonal
                            row = np.append(row,indVec[tris[e-1,j-1]-1]-1)
                            col = np.append(col,indVec[tris[e-1,i-1]-1]-1)
                            data = np.append(data,k_val)
    K = sp.csr_array((data, (row.astype(int), col.astype(int))))
    return K

# Assemble global load vector
def loadF2d(k,f,g,h,triangles,nodes,edges,triangleMidPts,midNodes,indVec,Gamma1Nodes, 
            Gamma1MidNodes,Gamma2Edges,shapeFn,noOfIntegPt,noOfIntegPt1d):
    F = np.zeros(np.count_nonzero(indVec))
    if shapeFn == 1:
        tris = triangles
    elif shapeFn ==2:
        tris = np.hstack((triangles, triangleMidPts))
    for e in range(1, np.shape(tris)[0]+1):
        for i in range(1, 3*shapeFn+1):
            if indVec[tris[e-1,i-1]-1] != 0:
                F[indVec[tris[e-1,i-1]-1]-1] += fei2d(k,f,g,h,e,i,triangles,nodes,edges,triangleMidPts,midNodes,Gamma1Nodes,
                                                      Gamma1MidNodes,Gamma2Edges,shapeFn,noOfIntegPt,noOfIntegPt1d)[0]
    return F

def rcm(A):
    perm = sp.csgraph.reverse_cuthill_mckee(A)
    perm_inv = np.copy(perm)
    perm_inv[perm] = np.arange(len(perm))
    return perm, perm_inv


# In[8]:


# Interpolated FEM solution function
def approxSol2d(w,g,indVec,nodes,triangles,midNodes,triangleMidPts,shapeFn):

    if (shapeFn == 1): # if linear shape functions are used
        uh = lambda x,y: linearApprox2d(x,y,w,g,indVec,nodes,triangles)
    else:    # if quadratic shape functions are used
        uh = lambda x,y: quadraticApprox2d(x,y,w,g,indVec,nodes,triangles,midNodes,triangleMidPts)
    
    return uh

# assemble linear approximated function
def linearApprox2d(x,y,w,g,indVec,nodes,triangles):

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    noNodes = nodes.shape[0]

    # put all node values together
    uu = np.zeros(noNodes)
    uu[indVec > 0] = w

    # Gamma 1 nodes
    gamma1s = (indVec==0)
    uu[gamma1s] = g(nodes[gamma1s,0],nodes[gamma1s,1])

    # vertices of triangles
    N1 = triangles[:,0]
    N2 = triangles[:,1]
    N3 = triangles[:,2]

    # number of triangles
    notri = len(N1)

    lxr,lxc = x.shape if x.ndim == 2 else (x.size, 1)
    lx =  lxr * lxc
    z = np.zeros(lx)
    
    flat_x = x.ravel()
    flat_y = y.ravel()

    ckNan = np.zeros(lx)
    for i in range(notri):
    
        x1, y1 = nodes[N1[i]-1]
        x2, y2 = nodes[N2[i]-1]
        x3, y3 = nodes[N3[i]-1]

        psi1 = shapeFn2d(1,flat_x,flat_y,x1,y1,x2,y2,x3,y3,1)
        psi2 = shapeFn2d(2,flat_x,flat_y,x1,y1,x2,y2,x3,y3,1)
        psi3 = shapeFn2d(3,flat_x,flat_y,x1,y1,x2,y2,x3,y3,1)

        ckNan += ((psi1!=0) | (psi2!=0) | (psi3!=0)).astype(float)

        z +=  uu[N1[i]-1] * psi1 + uu[N2[i]-1] * psi2 + uu[N3[i]-1] * psi3
        

        
    z[ckNan.astype(bool)] /= ckNan[ckNan.astype(bool)]
    z[~ckNan.astype(bool)] = np.nan

    z = z.reshape(x.shape)

    return z


# assemble quadratic approximations
def quadraticApprox2d(x,y,w,g,indVec,nodes,triangles,midNodes,triangleMidPts):

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    noVer = nodes.shape[0]
    noNodes = noVer + midNodes.shape[0]

    # put all node values together
    uu = np.zeros(noNodes)
    uu[indVec > 0] = w
    gamma1s = np.where(indVec == 0)[0]
    gamma1nodes = gamma1s[gamma1s < noVer]
    gamma1midNodes = np.setdiff1d(gamma1s,gamma1nodes)

    uu[gamma1nodes] = g(nodes[gamma1nodes, 0], nodes[gamma1nodes, 1])
    uu[gamma1midNodes] = g(midNodes[gamma1midNodes-noVer, 0], midNodes[gamma1midNodes-noVer, 1])


    # all vertices and mid-edge nodes
    N1 = triangles[:,0]
    N2 = triangles[:,1]
    N3 = triangles[:,2]
    N4 = triangleMidPts[:,0]
    N5 = triangleMidPts[:,1]
    N6 = triangleMidPts[:,2]

    notri = len(N1)

    lxr, lxc = x.shape if x.ndim == 2 else (x.size, 1)

    lx = lxr * lxc
    z = np.zeros(lx)

    flat_x = x.ravel()
    flat_y = y.ravel()


    ckNan = np.zeros(lx)
    for i in range(notri): 

        x1, y1 = nodes[N1[i]-1]
        x2, y2 = nodes[N2[i]-1]
        x3, y3 = nodes[N3[i]-1]

        psi1 = shapeFn2d(1,flat_x,flat_y,x1, y1, x2, y2, x3, y3, 2)
        psi2 = shapeFn2d(2,flat_x,flat_y,x1, y1, x2, y2, x3, y3, 2)
        psi3 = shapeFn2d(3,flat_x,flat_y,x1, y1, x2, y2, x3, y3, 2)
        psi4 = shapeFn2d(4,flat_x,flat_y,x1, y1, x2, y2, x3, y3, 2)
        psi5 = shapeFn2d(5,flat_x,flat_y,x1, y1, x2, y2, x3, y3, 2)
        psi6 = shapeFn2d(6,flat_x,flat_y,x1, y1, x2, y2, x3, y3, 2)

        ckNan += ((psi1!=0) | (psi2!=0) | (psi3!=0) | (psi4!=0) | (psi5!=0) | (psi6!=0)).astype(float)
        z +=  uu[N1[i]-1] * psi1 + uu[N2[i]-1] * psi2 + uu[N3[i]-1] * psi3 + uu[N4[i]-1] * psi4 + uu[N5[i]-1] * psi5 + uu[N6[i]-1] * psi6


    z[ckNan.astype(bool)] /= ckNan[ckNan.astype(bool)]
    z[~ckNan.astype(bool)] = np.nan
  

    z = z.reshape(x.shape)

    return z
        

# takes the values of the nodes and gives the approximated solution gradient on the domain
# Gradient of interpolated FEM solution
def approxSolGrad2d(w,g,indVec,nodes,triangles,midNodes,triangleMidPts,shapeFn):

    if (shapeFn == 1): # if linear shape functions are used
        graduh = lambda x,y: gradLinearApprox2d(x,y,w,g,indVec,nodes,triangles);
    else:    # if quadratic shape functions are used
        graduh = lambda x,y: gradQuadraticApprox2d(x,y,w,g,indVec,nodes,triangles,midNodes,triangleMidPts);

    return graduh



# assemble the gradient of the linear shape functions on the whole domain
def gradLinearApprox2d(x,y,w,g,indVec,nodes,triangles):

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    noNodes = nodes.shape[0]

    # put all node values together
    uu = np.zeros(noNodes)
    uu[indVec > 0] = w

    # Gamma 1 nodes
    gamma1s = (indVec==0)
    uu[gamma1s] = g(nodes[gamma1s,0],nodes[gamma1s,1])

    # vertices of triangles
    N1 = triangles[:,0]
    N2 = triangles[:,1]
    N3 = triangles[:,2]

    # no of triangles
    notri = len(N1)

    lxr,lxc = x.shape if x.ndim == 2 else (x.size, 1)
    lx =  lxr * lxc
    gradz = np.zeros((lx, 2), dtype=float)
    
    flat_x = x.ravel()
    flat_y = y.ravel()


    ckNan = np.zeros(lx, dtype = float)
    for i in range(notri):
        x1, y1 = nodes[N1[i]-1]
        x2, y2 = nodes[N2[i]-1]
        x3, y3 = nodes[N3[i]-1]
    
        gradpsi1 = shapeFnGrad2d(1,flat_x,flat_y,x1,y1,x2,y2,x3,y3,1).T
        gradpsi2 = shapeFnGrad2d(2,flat_x,flat_y,x1,y1,x2,y2,x3,y3,1).T
        gradpsi3 = shapeFnGrad2d(3,flat_x,flat_y,x1,y1,x2,y2,x3,y3,1).T

        ckNan += np.any((gradpsi1 != 0) | (gradpsi2 != 0) | (gradpsi3 != 0), axis=1).astype(float)

        gradz += (uu[N1[i]-1] * gradpsi1 + uu[N2[i]-1] * gradpsi2 + uu[N3[i]-1] * gradpsi3)
    
    gradz[ckNan.astype(bool)] /= ckNan[ckNan.astype(bool), np.newaxis]
    gradz[~ckNan.astype(bool)] = np.nan

   
    return np.squeeze(gradz.reshape(lxr,lxc,2))


# assemble gradient of the quadratic shape functions on the whole domain
def gradQuadraticApprox2d(x,y,w,g,indVec,nodes,triangles,midNodes,triangleMidPts):

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    
    noVer = nodes.shape[0]
    noNodes = noVer + midNodes.shape[0]

    # put all node values together
    uu = np.zeros(noNodes)
    uu[indVec > 0] = w
    gamma1s = np.where(indVec == 0)[0]
    gamma1nodes = gamma1s[gamma1s < noVer]
    gamma1midNodes = np.setdiff1d(gamma1s,gamma1nodes)

    uu[gamma1nodes] = g(nodes[gamma1nodes, 0], nodes[gamma1nodes, 1])
    uu[gamma1midNodes] = g(midNodes[gamma1midNodes-noVer, 0], midNodes[gamma1midNodes-noVer, 1])


    # all vertices and mid-edge nodes
    N1 = triangles[:,0]
    N2 = triangles[:,1]
    N3 = triangles[:,2]
    N4 = triangleMidPts[:,0]
    N5 = triangleMidPts[:,1]
    N6 = triangleMidPts[:,2]

    notri = len(N1)

    lxr, lxc = x.shape if x.ndim == 2 else (x.size, 1)

    lx = lxr * lxc
    gradz = np.zeros((lx,2),dtype=float)

    flat_x = x.ravel()
    flat_y = y.ravel()


    ckNan = np.zeros(lx, dtype = float)
    for i in range(notri):
        x1, y1 = nodes[N1[i]-1]
        x2, y2 = nodes[N2[i]-1]
        x3, y3 = nodes[N3[i]-1]
    
        gradpsi1 = shapeFnGrad2d(1,flat_x,flat_y,x1,y1,x2,y2,x3,y3,2).T
        gradpsi2 = shapeFnGrad2d(2,flat_x,flat_y,x1,y1,x2,y2,x3,y3,2).T
        gradpsi3 = shapeFnGrad2d(3,flat_x,flat_y,x1,y1,x2,y2,x3,y3,2).T
        gradpsi4 = shapeFnGrad2d(4,flat_x,flat_y,x1,y1,x2,y2,x3,y3,2).T
        gradpsi5 = shapeFnGrad2d(5,flat_x,flat_y,x1,y1,x2,y2,x3,y3,2).T
        gradpsi6 = shapeFnGrad2d(6,flat_x,flat_y,x1,y1,x2,y2,x3,y3,2).T

        ckNan += np.any((gradpsi1 != 0) | (gradpsi2 != 0) | (gradpsi3 != 0) | (gradpsi4 != 0) | (gradpsi5 != 0) | (gradpsi6 != 0), axis=1).astype(float)
        gradz += (uu[N1[i]-1] * gradpsi1 + uu[N2[i]-1] * gradpsi2 + uu[N3[i]-1] * gradpsi3 + uu[N4[i]-1] * gradpsi4 + uu[N5[i]-1] * gradpsi5 + uu[N6[i]-1] * gradpsi6)
    
    gradz[ckNan.astype(bool)] /= ckNan[ckNan.astype(bool), np.newaxis]
    gradz[~ckNan.astype(bool)] = np.nan


    return np.squeeze(gradz.reshape(lxr,lxc,2))


# In[9]:


def triangleMidPts(nodes, triangles, edges):
    nnodes = np.shape(nodes)[0]
    triangleMidPts = np.copy(triangles)
    for i in range(np.shape(triangles)[0]):
        triangleMidPts[i,0] = np.where((edges == np.sort(triangles[i,[0,1]])).all(axis=1))[0][0]
        triangleMidPts[i,1] = np.where((edges == np.sort(triangles[i,[1,2]])).all(axis=1))[0][0]
        triangleMidPts[i,2] = np.where((edges == np.sort(triangles[i,[0,2]])).all(axis=1))[0][0]
    triangleMidPts += nnodes+1
    return triangleMidPts

def midNodes(edges, nodes):
    midNodes = np.copy(edges)
    midNodes = np.hstack(((nodes[edges[:,0]-1][:,0]+nodes[edges[:,1]-1][:,0]).reshape(-1,1), 
                          (nodes[edges[:,0]-1][:,1]+nodes[edges[:,1]-1][:,1]).reshape(-1,1)))/2
    return midNodes

def bdyNodeEdge(nodes, edges, triangles):
    triedges = np.vstack((triangles[:,[0,1]], triangles[:,[1,2]], triangles[:,[0,2]]))
    bdyEdge = np.zeros(np.shape(edges)[0]).astype(bool)
    for i in range(np.shape(edges)[0]):
        bdyEdge[i] = (np.shape(np.where((triedges == edges[i,:]).all(axis=1))[0])[0] == 1)
    bdyNodes = np.zeros(np.shape(nodes)[0]).astype(bool)
    bdyNodes[np.unique(edges[np.argwhere(bdyEdge).flatten(),:].flatten())-1] = True
    return bdyNodes, bdyEdge

def Gamma1Nodes(nodes):
    Gamma1Nodes = (-1-1e-12<nodes[:,0])*(nodes[:,0]<1+1e-12)*(1-1e-12<nodes[:,0]**4+nodes[:,1])*(nodes[:,0]**4+nodes[:,1]<1+1e-12)
    return Gamma1Nodes

def Gamma2Edges(nodes, edges, bdyEdge):
    #Gamma2 = (x=-2, 0<y<2), (x=2, 0<y<2), (-2<x<2, y=2), (-2<x<-1, y=0), (1<x<2, y=0)
    Gamma2Nodes = (-2-1e-12<nodes[:,0])*(nodes[:,0]<-2+1e-12)*(-1e-12<nodes[:,1])*(nodes[:,1]<2+1e-12)+\
                    (2-1e-12<nodes[:,0])*(nodes[:,0]<2+1e-12)*(-1e-12<nodes[:,1])*(nodes[:,1]<2+1e-12)+\
                    (-2-1e-12<nodes[:,0])*(nodes[:,0]<2+1e-12)*(2-1e-12<nodes[:,1])*(nodes[:,1]<2+1e-12)+\
                    (-2-1e-12<nodes[:,0])*(nodes[:,0]<-1+1e-12)*(-1e-12<nodes[:,1])*(nodes[:,1]<1e-12)+\
                    (1-1e-12<nodes[:,0])*(nodes[:,0]<2+1e-12)*(-1e-12<nodes[:,1])*(nodes[:,1]<1e-12)
    Gamma2Edges = Gamma2Nodes[edges[:,0]-1]*Gamma2Nodes[edges[:,1]-1]*bdyEdge
    return Gamma2Edges

def Gamma1MidNodes(nodes, edges, bdyEdge):
    Gamma1MidNodes = Gamma1Nodes(nodes)[edges[:,0]-1]*Gamma1Nodes(nodes)[edges[:,1]-1]*bdyEdge
    return Gamma1MidNodes

def indVec(Gamma1Nodes, Gamma1MidNodes, shapeFn):
    if shapeFn==2:
        nodesbool = np.concatenate((Gamma1Nodes, Gamma1MidNodes))
    else:
        nodesbool = Gamma1Nodes
    indVec = np.zeros(len(nodesbool))
    count = 1
    for inode in range(len(nodesbool)):
        if not nodesbool[inode]:
            indVec[inode] = count
            count+=1
    indVec = np.array([int(indVeci) for indVeci in indVec])
    return indVec

def curveEdge(edges, Gamma1Nodes):
    curveEdge = np.zeros(np.shape(edges)[0]).astype(bool)
    curveEdge = Gamma1Nodes[edges[:,0]-1]*Gamma1Nodes[edges[:,1]-1]
    return curveEdge

# Compute H1 norm of error
def H1norm2d(ef, efgrad, nodes, triangles):
    error = 0.0
    def integrand(x, y):
        e = ef(x, y)
        grad_e = efgrad(x, y)
        return e**2 + (grad_e**2)[:,0] + (grad_e**2)[:,1]
    triNodes = np.hstack((nodes[triangles[:,0] - 1], nodes[triangles[:,1] - 1], nodes[triangles[:,2] - 1]))
    x1 = triNodes[:,0]
    y1 = triNodes[:,1]
    x2 = triNodes[:,2]
    y2 = triNodes[:,3]
    x3 = triNodes[:,4]
    y3 = triNodes[:,5]
    error += np.sum(quad2dTri(integrand, x1, y1, x2, y2, x3, y3, noOfIntegPt=7))
    return np.sqrt(error)