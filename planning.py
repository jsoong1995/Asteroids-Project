import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random
import math
import time
import csv

'''
4/17/2021:
    1) Added goal class for plotting purposes
    2) Modified "build_cspace" with ability to pass goal game object
    3) Added command to plot goal coordinates.

5/9/2021:
    1) implemented RRT with existing kinodynamics








'''



def motion_planning(spacecraft,goalZone,asteroid_list,fps,screen_width,screen_height): # Eric Added Full Function
    # Time Stepping Settings
    time_limit = 20 # seconds
    time_inc = 1/fps # second

    obs_list = []
    print('Initializing List of Obstacle Classes')

    for ast in asteroid_list:
        print(f'Asteroid Radius: {ast.radius}')
        obs_list.append(obstacle(ast.position,ast.velocity,ast.radius,fps,time_limit,screen_width,screen_height))

    # Assemble Start Node Structure from Spacecraft Attributes
    start_theta = math.atan2(spacecraft.direction[1],spacecraft.direction[0])*(180/np.pi)
    print(f'Start Angle = {start_theta}')
    startNode = myNode(1,spacecraft.position[0],spacecraft.position[1],0.0,start_theta,spacecraft.velocity[0],spacecraft.velocity[1],0,0)

    # Define Spaceship Radius
    vehicle_rad = spacecraft.radius
    print(f'Spaceship Radius: {vehicle_rad}')

    # Define Spaceship Turning Velocity [negative, zero, positive]
    vehicle_w = [-spacecraft.MANEUVERABILITY, 0, spacecraft.MANEUVERABILITY]

    # Define Spaceship Acceleration [negative, zero, positive]
    vehicle_acc = [-spacecraft.ACCELERATION, 0, spacecraft.ACCELERATION]

    # Assemble Goal Region Structure
    gl = myGoal(goalZone.position[0],goalZone.position[1],goalZone.radius)
    print(f'Goal Radius: {goalZone.radius}')

    # Define Epsilon
    epsilon = 3.0

    # Define Time Boundaries
    t_bounds = [0, time_limit]

    # Open Figure Axes
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Establish that Goal is NOT Found
    foundGoal = False

    # Start Timing the Receding Horizon Algorithm
    start_time = time.time()

    # Run Receding Horizon Loop
    # while not foundGoal:
    for _ in range(10):
        print('RUNNING NEW HORIZON WINDOW')

        # Execute RRT Algorithm
        solve = myRRT3d(startNode,gl,epsilon,screen_width,screen_height,t_bounds,obs_list,time_inc,vehicle_rad,vehicle_acc,vehicle_w,ax)

        # Write to Output File
        writeMovementFile(solve,'results.txt')

        # Write Inputs File
        writeInputsFile(solve,'inputs.txt')

        # Check if Goal was Found Status
        foundGoal = solve.status

        # Establish End of Previous Search as Start Node for Next Search
        final_ID = solve.finalPathIDs[-1]
        finalNode = solve.searchedNodes[final_ID-1]
        startNode = myNode(1,finalNode.x,finalNode.y,finalNode.t,finalNode.theta,finalNode.vel_x,finalNode.vel_y,finalNode.u_a,finalNode.u_w) # Start Node ID is 1 (First Node))

        # Reset Z_Bounds to Not Search Below New Start Node
        t_bottom = finalNode.t
        t_bounds[0] = t_bottom


    # End the timer for the Receding Horizon Algorithm
    end_time = time.time()

    # Print Time of the Algorithm to the Screen
    runtime = end_time - start_time
    print(f'Runtime of Receding Horizon Algorithm: {runtime}')

    # Show Figure
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Time')
    ax.view_init(90,0)
    # ax.set_xlim(98,102)
    # ax.set_ylim(448,452)
    plt.show()

    '''


# John-added command
    for i in range(fps*time_limit):
        ax.plot3D(gl.xpos[i],gl.ypos[i],obs.t[i],marker='o',c=colors[1])

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Time')
    ax.view_init(30,45)
    plt.xlim(0,screen_width)
    plt.ylim(0,screen_height)

    plt.show()
    '''
# obstacle class --------------------------------------------------------------
class obstacle(): # Eric Added Full Class
    def __init__(self, pos, vel, radius, fps, time_lim, screen_width, screen_height):
        self.x = np.array([pos[0]])
        self.y = np.array([pos[1]])
        self.t = np.linspace(0,time_lim,fps*time_lim + 1)
        self.rad = radius
        self.fps = fps
        self.time_lim = time_lim
        self.screen_width = screen_width
        self.screen_height = screen_height

        for _ in range(self.t.size-1):
            self.x = np.append(self.x, (self.x[-1]+vel[0]) % screen_width)
            self.y = np.append(self.y, (self.y[-1]+vel[1]) % screen_height)


    def collisionCheck(self,x,y,t,rVehicle):

        # Find the time at which end node occurs
        endTime = t
        # Find index in which endTime occurs in self.t
            # Tolerance is added since times intervals aren't exactly the same
            # np.where returns a tuple (array of indices,datatype)
            # [0][0] references the first element of tuple, first element of array
        timeIndex = np.where(abs(self.t - endTime) < 0.01)[0][0]
        # Find obstacle coordinates at time instance for collision checking
        xObstInstance = self.x[timeIndex]
        yObstInstance = self.y[timeIndex]
        if ((x - xObstInstance)**2 + (y - yObstInstance)**2)**0.5 <= 1.2*rVehicle + self.rad:
            return True
        else:
            return False

    def plotMe(self,ax,color):

        # Plot Obstacle Trajectory as a Line
        xpoints = self.x
        ypoints = self.y
        zpoints = self.t

        # Line Width
        lw = math.trunc(self.rad/5)

        for slc in unlink_wrap(xpoints, ypoints, [0, self.screen_width], [0, self.screen_height]):
            print(slc)
            ax.plot3D(xpoints[slc],ypoints[slc],zpoints[slc],'-',linewidth=lw,c=color)

# unlink

def unlink_wrap(xdat, ydat, xlims = [0, 100], ylims = [0, 100], thresh = 0.95):
    """
    Full Citation:
    farenorth. (2014, Nov 26). Preventing plot joining when values
        “wrap” in matplotlib plots. Stack Overflow. https://stackoverflow.com/
        questions/27138751/preventing-plot-joining-when-values-wrap-in-matplotlib-plots

    Iterate over contiguous regions of `dat` (i.e. where it does not
    jump from near one limit to the other).

    This function returns an iterator object that yields slice
    objects, which index the contiguous portions of `dat`.

    This function implicitly assumes that all points in `dat` fall
    within `lims`.

    """
    xjump = np.nonzero( np.abs(np.diff(xdat)) > ((xlims[1] - xlims[0]) * thresh) )[0]
    yjump = np.nonzero( np.abs(np.diff(ydat)) > ((ylims[1] - ylims[0]) * thresh) )[0]

    jumps = np.append(xjump, yjump)
    print(jumps)
    sorted_jumps = np.sort(jumps)
    print(sorted_jumps)

    lasti = 0
    for ind in sorted_jumps:
        yield slice(lasti, ind + 1)
        lasti = ind + 1
    yield slice(lasti, len(xdat))

    '''
    lasti = 0
    for ind in xjump:
        yield slice(lasti, ind + 1)
        lasti = ind + 1
    #yield slice(lasti, len(xdat))

    lasti = 0
    for ind in yjump:
        yield slice(lasti, ind + 1)
        lasti = ind + 1

    yield slice(lasti, len(ydat))
    '''
################################################################################
'''
class goal(): # John-added full class
    def __init__(self,pos,size,fps,time_lim,screen_width,screen_height):
        self.xpos = np.array([pos[0]])
        self.ypos = np.array([pos[1]])
        self.t = np.linspace(0,time_lim,fps*time_lim)
        self.rad = size

        for t in self.t:
            self.xpos = np.append(self.xpos,self.xpos[-1] % screen_width)
            self.ypos = np.append(self.ypos,self.ypos[-1] % screen_height)
'''
# Writing position coordinates ------------------------------------------------
'''
- takes in data structure "search" that contains node and path data.
- writes the x, y, coordinates out into a txt file that will be blitted
  onto the game screen
'''

def writeMovementFile(search,filename):
    with open(filename, 'a', newline='') as movefile:
        writer = csv.writer(movefile, delimiter=',')

        # writer.writerow(['New Horizon Window'])

        for id in search.finalPathIDs: # For Each ID in Final Path List...
            if not id == 1:
                node = search.searchedNodes[id-1]
                writer.writerow([node.x, node.y]) # Write xpos, ypos to txt File

    return

# Writing inputs --------------------------------------------------------------
'''
- takes in an RRT data structure and writes the translational acceleration
  and angular velocity inputs into a .txt file
- .txt file is then read to input handling for vehicle control
'''

def writeInputsFile(search,filename):
    pass

################################################################################
# RRT Algorithm Definition - Use Class To Access Stored Path and Tree

# RRT data structure ----------------------------------------------------------
'''
- data structure so found values can be accessed
- performs RRT and keeps path data
'''
class myRRT3d():
    def __init__(self,startNode,goalRegion,epsilon,screen_width,screen_height,z_bounds,obstacleList,z_inc,vehicle_rad,vehicle_acc,vehicle_w,ax):
        self.finalPathIDs = [] # List To Contain Solved Path Node IDs
        self.searchedEdges = [] # List to hold all Searched Edge Objects
        self.searchedNodes = [] # List to hold all Searched Node Objects
        self.status = False # False = Goal NOT Found; True = Goal Found

        # Number of Paths that Reach Time Horizon
        counter = 0
        num_paths = 10

        # Establish Time Horizon Window
        current_time = z_bounds[0]
        time_horizon = 2.0 # second

        # Assemble CSpace Limits
        cSpaceX = [0, screen_width]
        cSpaceY = [0, screen_height]
        cSpaceZ = z_bounds

        # Plot Obstacles
        print('Plotting Obstacles')
        colors = ('b','g','r','c','m','y')
        for obs in obstacleList:
            obs.plotMe(ax,colors[obstacleList.index(obs)])

        # Initialize Empty Search Tree
        T = myTree()

        # Add Start Node to Search Tree
        T.addNode(startNode)
        print(f'***** Start Node at X: {startNode.x} Y: {startNode.y} Z: {startNode.t} *****')

        # Plot Start Node on Figure
        startNode.plotMe(ax)

        # Plot Goal Region on Figure
        goalRegion.plotMe(ax,cSpaceZ)

        # Keep Track of Number of Nodes in Tree with new_ID counter. Initialize counter
        # at 2 (startNode has ID 1 already). Add 1 to ID counter with each new Node
        # that is added to Tree (below)
        new_id = 2

        # Run RRT Expanding Search Until End of Horizon Window
        # for _ in range(1000):
        while counter < num_paths:
            # Randomly Sample a Point in C-Space and Create "Dummy" Node u
            u = randomNode(cSpaceX,cSpaceY)
            print(f'Random Node at X: {u.x} Y: {u.y} Z: {u.t}')

            # Identify Closest Node v in Tree to Randomly Sampled Node u
            v = closestNode(u,T.nodeList)
            print(f'Node {v.id} is Closest in Tree, at X: {v.x} Y: {v.y} Z: {v.t}')

            # Choose Random Acceleration and Turning Inputs
            # u_a = random.choice(vehicle_acc) # Pick Random a from List
            # u_w = random.choice(vehicle_w) # Pick Random w from List

            # Create Node w Extended from v Based on Vehicle Dynamics

            # Creating valid trajectory edge
            edgeValid = pathHunt(v,u,new_id,obstacleList,z_inc,vehicle_rad)
            if edgeValid == None:
                continue
            w = edgeValid.endNode
            passVar = False
            for leaf in T.nodeList:
                if w.x == leaf.x and w.y == leaf.y:
                    passVar = True
                    break
            if passVar == True:
                continue
            print(f'Epsilon Node Generated at X: {w.x} Y: {w.y} Z: {w.t}')
            #time.sleep(0.5)
            tol = 1e-3
            if w.t <= z_bounds[0]+time_horizon+tol:
                # If Edge NOT Intersecting with any obstacle, Add Node w to the Search Tree
                T.addNode(w)
                print(f'----- Node {w.id} Added to Search Tree. Not in any Obstacle.')
                # Update Number of Times the Horizon Window is Reached
                if (w.t < z_bounds[0]+time_horizon+tol) and (w.t > z_bounds[0]+time_horizon-tol):
                    counter += 1
                # Plot Node w on Figure
                w.plotMe(ax)
                # Define Node v as Parent to Node w
                w.parentNode = v
                print(f'----- Node {w.id} now has Parent Node {v.id}')
                # Add Edge Connecting Node v and Node w to Search Tree
                T.addEdge(edgeValid)
                print(f'----- Edge Created Between Node {edgeValid.startNode.id} and Node {edgeValid.endNode.id}')
                # Plot edgeVW on Figure
                edgeValid.plotMe(ax)
                # Increase new_id Counter (as we have added previous new_id to a Node w just now)
                new_id += 1

            # Check if Node w is in Goal Region
            if goalRegion.check(w):
                print(f'Reached Goal! RRT Algorithm Complete.')
                print(f'Final Node ID: {w.id} X: {w.x} Y: {w.y}. Goal at X: {goalRegion.x} Y: {goalRegion.y}.')
                self.searchedNodes = T.nodeList # List of ALL Node Objects in Tree
                self.searchedEdges = T.edgeList # List of ALL Edge Objects in Tree
                # First Node in Search Tree (index 0) is the Start of Path.
                # Final Node in Search Tree (index is numbder of nodes in list minus 1) is the End of Path.
                self.finalPathIDs = self.givePath(T.nodeList[0],T.nodeList[T.numNodes-1]) # List of Node IDs that Trace Path from Start to Goal
                # Plot Path in Red
                self.plotPath(ax)
                # Status of Goal Search = Goal Found
                self.status = True
                return

        # After Max Iterations Reached... Return Path That Get's Closest to the Goal
        print(f'Goal Not Found Yet.')
        print(f'Returning Path Closest to Goal.')
        self.searchedNodes = T.nodeList # List of ALL Node Objects in Tree
        self.searchedEdges = T.edgeList # List of ALL Edge Objects in Tree
        # First Node in Search Tree (index 0) is the Start of Path.
        # Identify Closest Node to Goal Region as the End of Path
        self.finalPathIDs = self.givePath(T.nodeList[0],goalRegion.nearestNode(T.nodeList))
        # Plot Path in Red
        self.plotPath(ax)
        return

    def givePath(self,startNode,goalNode):
        # Construct Inverse Path in List (Path IN REVERSE) and then
        # REVERSE the Inverse Path to correct Forward Path before returning
        path = [] # Container For Path of Node IDs
        path.append(goalNode.id) # Add Goal Node ID to Inverse Path List
        prev_node = goalNode.parentNode # Call Parent of Goal Node to Begin Back Tracking to Start Node

        while not prev_node == 0: # Check if prev_node is Start Node
            path.append(prev_node.id) # Add prev_node ID to Inverse Path
            prev_node = prev_node.parentNode # Steps Back Along Path

        path.append(startNode.id) # While Loop Ended when Start Node was Reached, Still Need to Add ID to Path

        path.reverse() # Reverse the Inverse Path to Forward Path

        return path

    def plotPath(self,ax):
        # Plot Path in RED
        xpoints = []
        ypoints = []
        zpoints = []
        for id in self.finalPathIDs:
            plotnode = self.searchedNodes[id-1]
            xpoints.append(plotnode.x)
            ypoints.append(plotnode.y)
            zpoints.append(plotnode.t)
        ax.plot3D(xpoints,ypoints,zpoints,'r-',linewidth=2)

# Path Hunting function -------------------------------------------------------
'''
- "pathHunt" takes in a start node, end node, epsilon, next idm and obstacles
- A bunch of feasible paths are created and iterated over the entirety of the
range of permissible acceleration values.
- The path will be generated using a "traj" function.
- The path information containing the shortest distance between its last point
and end node.
- Should return an edge data structure
- Print statements should be commented out as they unnecessarily clog the
console
'''

def pathHunt(nodeStart,nodeRand,newID,obstacles,tDelta,rVehicle):
    minDist = 10**3
    edgeBest = None
    a = [-0.25,0,0.25] # in pixels/frame
    omega = [-math.pi/6, 0 ,math.pi/6] # in radians/frame
    for i in range(0,len(a)):
        for j in range(0,len(omega)):
            edgePossible = traj(newID,nodeStart,a[i],omega[j],obstacles,tDelta,rVehicle)
            if edgePossible != None:
                dist = ((edgePossible.endNode.x - nodeRand.x)**2 + (edgePossible.endNode.y - nodeRand.y)**2)**0.5
                if dist < minDist:
                    minDist = dist
                    edgeBest = edgePossible
                    print(f'Edge Best - Acceleration: {edgeBest.endNode.u_a} Omega: {edgeBest.endNode.u_w} ')

    return edgeBest

# Trajectory function ---------------------------------------------------------

def traj(newID,nodeStart,accel,angVel,obstacles,timeStep,rVehicle):

    # Time Step For Each Edge in Trajectory
    dt = timeStep
    # Vehicle Inputs
    a = accel
    omg = angVel
    # Initial State Before Trajectory
    tht = [nodeStart.theta]
    vx = [nodeStart.vel_x]
    vy = [nodeStart.vel_y]
    x = [nodeStart.x]
    y = [nodeStart.y]
    # Start Time at Beginning of Trajectory
    t = [nodeStart.t]

    # Generate Trajectory 5 Time Steps into the Future
    num_steps = 10

    trajectoryGenerated = []

    for _ in range(num_steps):
        tht.append(tht[-1] + omg)
        vx.append(vx[-1] + a*math.cos(tht[-1]*(math.pi/180)))
        vy.append(vy[-1] + a*math.sin(tht[-1]*(math.pi/180)))
        x.append(x[-1] + vx[-1])
        y.append(y[-1] + vy[-1])
        t.append(t[-1] + dt)

        # print(f'The trajectory parameters:{a},{omg},{tht},{vx},{vy},{x},{y}')
        inObst = False
        for obst in obstacles:
            if obst.collisionCheck(x[-1],y[-1],t[-1],rVehicle) == True:
                inObst = True
                break

    if inObst == False:
        nodeEnd = myNode(newID,x[-1],y[-1],t[-1],tht[-1],vx[-1],vy[-1],a,omg)
        edgeGenerated = myEdge(nodeStart,nodeEnd)
        edgeGenerated.trajX = x
        edgeGenerated.trajY = y
        edgeGenerated.trajT = t
        return edgeGenerated


################################################################################
# Randomly Generate a Node in the Defined C-Space

def randomNode(x_range,y_range):
    # x_range is Tuple of (xmin,xmax) of C-space
    # y_range is Tuple of (ymin,ymax) of C-space
    # z_range is Time Span of Simulation

    # Node Created with 0 ID, at Random X, Y and Z Location within C-Space
    return myNode(0,random.uniform(x_range[0],x_range[1]),random.uniform(y_range[0],y_range[1]),0,0,0,0,0,0)

################################################################################
# Calculates New Node from Tree Node based on Vehicle Dynamics


################################################################################
# Find Closest Node in Search Tree to Specified Node

def closestNode(sampledNode,nodeList):
    min_dist = 1e3 # Min Distance from Sampled Node to Closest Tree Nodes

    # Calculate Distance from Sampled Node to Each Node in List
    for treeNode in nodeList:
        # Distance from Sampled Node to Tree Node in List
        check_dist = math.sqrt((sampledNode.x-treeNode.x)**2 + (sampledNode.y-treeNode.y)**2)
        # Checked Distance is Smaller Than Current Minimum Distance
        if check_dist < min_dist:
            # Add Distance to List
            min_dist = check_dist
            # Identify the Index of the Closest Node
            min_index = treeNode.id - 1

    # Return Node at Minimum Distance to Sampled Node
    return nodeList[min_index]

################################################################################
# Expanding Search Tree Class Definition

class myTree():
    def __init__(self):
        self.numNodes = 0 # Number of Nodes in Tree
        self.numEdges = 0 # Number of Edges in Tree
        self.nodeList = [] # List of Node Objects, Index References Node Object
        self.edgeList = [] # List of Edge Objects, Index References Edge Object

    def addNode(self,node):
        self.nodeList.append(node)
        self.numNodes += 1

    def addEdge(self,edge):
        self.edgeList.append(edge)
        self.numEdges += 1

################################################################################
# Circular Goal Region Class definition

class myGoal():
    def __init__(self,xpos,ypos,radius):
        self.x = xpos
        self.y = ypos
        self.rad = radius

    def check(self,node):
        # Check if Given Node is Within Boundary of Goal Region
        # Distance from Node to Center of Goal
        dist2cen = math.sqrt((node.x-self.x)**2 + (node.y-self.y)**2)
        # Check if Distance from Node to Goal Center is less than Goal Region Radius
        if (dist2cen <= self.rad):
            return True # Node Inside Goal Region, Done Search
        else:
            return False # Node Outside Goal Region, Not Done Search

    def nearestNode(self,nodeList):
        # Find The Node that is Closest to the Goal at end of Horizon Window
        dist2goal = [] # Empty List to Hold Distances from Sampled Node to Tree Nodes
        # Calculate Distance from Goal to Each Node in List
        for treeNode in nodeList:
            # Distance from Goal to Tree Node in List
            check_dist = math.sqrt((self.x-treeNode.x)**2 + (self.y-treeNode.y)**2)
            # Add Distance to List
            dist2goal.append(check_dist)
        # Identify the Index of Minimum Distance in List
        min_index = dist2goal.index(min(dist2goal))
        # Return Node at Minimum Distance to Goal
        return nodeList[min_index]

    def plotMe(self,ax,cSpaceZ):
        # Height of Cylinder Goal Region
        plot_z = np.linspace(cSpaceZ[0],cSpaceZ[1],50)
        # Array of Angles around a Circle
        theta = np.linspace(0,2*np.pi,50)
        # ----- Not Sure What This Does ??? -----
        theta_grid, z_grid = np.meshgrid(theta, plot_z)
        # Calculate X and Y Values for A Circle around the Goal Center
        x_grid = self.x + self.rad*np.cos(theta_grid)
        y_grid = self.y + self.rad*np.sin(theta_grid)
        # Plot Cylindrical Surface
        ax.plot_surface(x_grid,y_grid,z_grid,alpha=0.9)

################################################################################
# Node Class Definition

class myNode():
    def __init__(self,id,xpos,ypos,t,theta,vel_x,vel_y,in_acc,in_omega):
        self.id = id # Numerical ID of Node
        self.x = xpos # Physical X Location of Vehicle Center
        self.y = ypos # Physical Y Location of Vehicle Center
        self.t = t # Time of Vehicle Center

        # Kinematic Elements
        self.theta = theta
        self.vel_x = vel_x
        self.vel_y = vel_y

        # Inputs Used Before this State
        self.u_a = in_acc
        self.u_w = in_omega

        # self.costToStart = 0.0 # Tracks Cost to Start Node, Initially Empty

        # self.status = 0 # Status 0 as Unvisited; Change to 1 when Visited
        self.parentNode = 0 # Identifies Parent Node Object for Final Path (Default as None - Start Node has No Parent)

        self.numOutgoingEdges = 0 # Number of Edges leaving from this Node
        self.outgoingEdges = [] # List of Edge Objects leaving from this Node

        self.numIncomingEdges = 0 # Number of Edges into this Node
        self.incomingEdges = [] # List of Edge Objects enterring this Node

    # Function to Set Visit Status to Open
    def visit(self):
        self.status = 1

    # Function to Plot Node on Figure
    def plotMe(self,ax):
        ax.plot3D(self.x,self.y,self.t,'bo',markersize=2)

################################################################################
# Edge Class Definition

class myEdge():
    def __init__(self,start_node,end_node):
        self.startNode = start_node # Node Object at Start of Edge
        self.endNode = end_node # Node Object at End of Edge
        self.edgeCost = math.sqrt((self.endNode.x-self.startNode.x)**2 \
            + (self.endNode.y-self.startNode.y)**2) # Cost to move from Start Node to End Node

        self.trajX = []
        self.trajY = []
        self.trajT = []

        # Add New Edge to Start Node Object
        start_node.outgoingEdges.append(self)
        start_node.numOutgoingEdges += 1

        # Add New Edge to End Node Object
        end_node.incomingEdges.append(self)
        end_node.numIncomingEdges += 1

    # Function to Plot Edge on Figure
    def plotMe(self,ax):
        xpoints = self.trajX
        ypoints = self.trajY
        zpoints = self.trajT
        ax.plot3D(xpoints,ypoints,zpoints,'k-',linewidth=1)

################################################################################
