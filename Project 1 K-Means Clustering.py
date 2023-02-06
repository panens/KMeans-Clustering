import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt 


def EuclidianDistance(x,y): #returns Euclidian distance for any amount of dimensions x and y have 
    #implies that x and y have the same number of dimensions
    k = len(x) #reads in the amount of dimensions of x
    distance = 0 #sets distance to 0
    for i in range(0,k,1): #for each dimension of x 
        
        r = (x[i]-y[i]) ** 2 # squares the difference between the value of x and y for same dimension
        
        distance = distance + r #keeps a running total of all distances for all dimensions. Ends as total of squared distances at the end of the loop
        
    
    return np.sqrt(distance) #returns the square of the total squared distances. 


def PickInitialCentroids(df, k): #picks random k centers from dataframe df
    print("---Entered Function Pick Initial Centroids---")
    z = df.shape[0] #saves how many rows the dataframe has in variable z
    print(z)
    result = random.sample(range(0,z),k) #picks a random sample of k elements from range 0-z without repetition
    print(result) 
    meansdf = df.iloc[result] #creates a df of only the selected means for calculations
    print(meansdf)
    print("---Leaving Function Pick Initial Centroids---")
    return meansdf

def InitialIteration(wdf, initial_centroids): #I separated the initial iteration from the following as I worked as it was easier to understand everything this way
    new_category = []
    for i, rowa in wdf.iterrows(): # for every element of the working dataframe wdf
    
        distances = [] #creates an empty list
        
        for j, rowb in initial_centroids.iterrows(): #iterates through all initial centroids
            #print(rowa)
            #print(rowb)
            distances.append(EuclidianDistance(rowa,rowb)) #calculates euclidian distance between each point and each centroid. 
            #and appends it to list distances. 
        
        #print(distances) #This was used during code development to test on the go
        #print(distances.index(min(distances))) #This was used during code development to test on the go
        new_category.append((distances.index(min(distances)))) #appends the index of the lowest distance to te category. 
        #this is essentially our categorization on which mean the dot belongs to
        
    return new_category

def PickNewCentroids(wdf,new_category): #after running the iteration witinitil centroids
    #we need to create new centroids
    wdf["Category"] = new_category #category assigned is appended to the working dataframe
    new_centroids = wdf.groupby(["Category"], as_index=False)[['CulmenLength_mm','FlipperLength_mm', 'BodyMass_g']].mean() #this row
    #groups the entire working dataframe by category which are 0, 1, 2 when k is 3. 
    #it calculates the mean within the category for each of the 3 variables of interest. 
    return new_centroids[['CulmenLength_mm','FlipperLength_mm', 'BodyMass_g']] 
    
        



def kMeans(wdf, k):
    initial_centroids = PickInitialCentroids(wdf,k) #picks k random initial centroids
    
    new_category = InitialIteration(wdf,initial_centroids) #runs the initial iteration and returns initial categorization
        
    print("Initial Categorization Complete")
    breaker = True #creates a variable breaker = True which will be used to keep the while loop going. 
    counter = 1 #just to track how many iterations are being done. 
    while breaker: # ensures code will be ran until breaker is assigned value false
        old_category = new_category #old category becomes the last result from new category 
        print(f"Iteration: {counter}") # just keeps track on the terminal as the code is ran
        new_centroids = PickNewCentroids(wdf,old_category) # new centroids are picked with entry parameters working dataframe wdf and old_category which is the result of new_category from a previous iteration. 
        print("New Centroids: ") #just for keeping track on the terminal
        print(new_centroids) #just for keeping track on the terminal 
        new_category = [] #instanciates an empty list new_category
        """within_cluster_sum_of_squares = [] """
        
        for i, rowa in wdf.iterrows(): #for every element of the working dataframe
        
            distances = [] #create an empty list 
            
            for j, rowb in new_centroids.iterrows(): # and compute the distance between each of the new centroids. 
                #print(rowa) #just used for debugging
                #print(rowb) #just used for debugging
                distances.append(EuclidianDistance(rowa[['CulmenLength_mm','FlipperLength_mm', 'BodyMass_g']],rowb[['CulmenLength_mm','FlipperLength_mm', 'BodyMass_g']]))
                #appends the distances to a distance variable so they can be used later for categorization
            
            #print(distances)
            #print(distances.index(min(distances)))
            new_category.append((distances.index(min(distances)))) #just as in the initial iteration, the index of te minimal distance is the new category for this point
            """within_cluster_sum_of_squares.append(min(distances)) #"""
        
        if old_category == new_category: # if there were no changes between categories between last iteration and the current one
            breaker = False #changing the breaker variable to false will break the while loop 
        else : #if there is at least one point that changed its categorization 
            counter = counter + 1 #we increase the counter and the while loop is entered again. 
    
    """tssk = sum(within_cluster_sum_of_squares)"""
    
    return new_centroids, new_category, counter #finally, when the while loop is broken, we want to retrieve the final new centroids, final new categories, and the final iteration counter. 

def WithinClusterSum(wdf,categories,centroids,k): 
    
    wdf ["Categories"] = categories
    newdf = pd.merge(wdf, centroids, how='inner', left_on = 'Categories', right_index = True)
    
    lista = []
    for i, row in newdf.iterrows():
        x = row[['CulmenLength_mm_x','FlipperLength_mm_x', 'BodyMass_g_x']]
        y = row[['CulmenLength_mm_y','FlipperLength_mm_y', 'BodyMass_g_y']]
        z = EuclidianDistance(x,y)
        lista.append(z)
     
    newdf["Result"] = lista
    
    newdf = newdf.groupby(["Categories"], as_index=False)[['Result']].sum()
    
    tssk = newdf["Result"].sum()
    
    return tssk

def autoNorm(wdf):
    minVals = wdf.min(0) # min values of all columns
    #print(minVals)
    maxVals = wdf.max(0) # max values of all columns
    #print(maxVals)
    ranges = maxVals - minVals # range is max minus min value for each column
    #print(ranges)
    normDataSet = np.zeros(np.shape(wdf)) # zero matrix size of working dataframe wdf matrix
    #print(normDataSet)
    m = wdf.shape[0] # amount of rows in wdf
    #print(m)
    normDataSet = wdf - np.tile(minVals, (m, 1)) # Actual value - minimum value
    #print(normDataSet)
    normDataSet = normDataSet/np.tile(ranges, (m, 1))  # Divides the previous value by the range to get a value between 0 and 1. 
    #print(normDataSet)
    
    #print statements just used to double check each step of the normalization process. 
    return normDataSet
    
    



def main() -> None: 
    
    list_of_k = []
    list_of_tssk = []
    list_of_iterrations = []
    u=10 # this is used within the code to specify the range for how many ks to run the code
    #as well as to make sure the ticks on the graph are correct and complete
    
    for i in range(u): 
        #Loading Penguing Data
        penguin_df = pd.read_csv(f'penguins.csv') 
        wdf = penguin_df[['CulmenLength_mm','FlipperLength_mm', 'BodyMass_g']]
        wdf = wdf.dropna()
        print(wdf)
    
        wdf = autoNorm(wdf) #normalizes data in all columns
        #if autoNorm row is hashed, the code will work with raw data instead. 
        
        centroids, categories, counter = kMeans(wdf,i+2)
        
        print("AFTER kMeans Function: ")
        print("Final Centroids: ")
        print(centroids)
        print("Categories: ")
        print(categories)
        print("Iterations: ")
        print(counter)
        
        
        tssk = WithinClusterSum(wdf,categories,centroids, i+2)
        print(f"----------RESULT{i}----------")
        print(tssk)
        print(f"----------RESULT{i}----------")
        
        list_of_k.append(i+2)
        list_of_tssk.append(tssk)
        list_of_iterrations.append(counter)
        
    result = pd.DataFrame({'k' : list_of_k, 'tssk' :list_of_tssk, 'iterrations' : list_of_iterrations})
    print(result)

    fig, ax = plt.subplots() # starts a plot
    ax.plot(result["k"],result["tssk"]) #graphs k on x axis and tssk on y axis 
    ax.set_title("The Elbow Method - Penguin Data") #titles the graph
    ax.set_ylabel("Within Group Sum of Squares") # yaxis label
    ax.set_xlabel("Number of Clusters") #xaxis label
    ax.grid(True) #shows grid 
    ax.scatter(result["k"],result["tssk"]) # shows points on graph

    for i, txt in enumerate(result["iterrations"]): #adds amount of iterrations next to each point
        txt = f"Iterrations: {txt}" 
        ax.annotate(txt, (result["k"][i],result["tssk"][i]))

    plt.xticks(np.arange(2,u+2,1))  
    plt.yticks(np.arange(10,100,10))
    plt.savefig("output.jpg")


if __name__ == "__main__": 
  main()