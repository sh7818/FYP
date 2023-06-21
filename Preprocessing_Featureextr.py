
    
def Criteria(data):
    Mat = data
    Y = well_detection(data)
    M = np.size(Mat,0)
    N = np.size(Mat,1)
    Criterion1 = 0
    Criterion2 = 0
    RoIs = [range(4,50),range(3,56)]
    RoI1 = [range(4,30),range(3,56)]
    RoI2 = [range(47,74),range(3,56)]

    for i in range(M):
        for j in range(N):
            if Y == 2:
                if ((i in RoI1[0]) or (i in RoI2[0])) and (j in RoI1[1]):
                    if Mat[i][j] < 50 :
                        pass
                    elif Mat[i][j] >= 50: 
                        Criterion1 = Criterion1 + 1
                else:
                    if Mat[i][j] < 50 :
                        pass
                    elif Mat[i][j] >= 50: 
                        Criterion2 = Criterion2 + 1
            elif Y == 1:
                if (i in RoIs[0]) and (j in RoIs[1]):
                    if Mat[i][j] < 50 :
                        pass
                    elif Mat[i][j] >= 50: 
                        Criterion1 = Criterion1 + 1
                else:
                    if Mat[i][j] < 50 :
                        pass
                    elif Mat[i][j] >= 50: 
                        Criterion2 = Criterion2 + 1
                
    return Criterion1,Criterion2

def Score(A,B):
    
    C1a,C2a = Criteria(A)
    C1b,C2b = Criteria(B)
    # print("A:",C1a,C2a)
    # print("B:",C1b,C2b)

    try:
        score = (C1b/C1a)+(C2a/C2b)     
    except: pass    
    return score


def Concentration_Cleaning(data,d,t):
    Mat = np.mean(data,axis=2)
    Raw = np.mean(data,axis=2)
    M = np.size(Mat,0)
    N = np.size(Mat,1)
    removed_pixels = []
    for m in range(M):
        for n in range(N):
            C = 0
            count_pixels = 0
            for i in range(round(m - (d+1)/2),round(m + (d+1)/2)):
                for j in range(round(n - (d+1)/2),round(n + (d+1)/2)):
                    try:
                        C = C + Mat[i][j]
                        count_pixels = count_pixels + 1
                    except: pass

            if C < t * count_pixels:
                removed_pixels.append([m,n])
                Mat[m][n] = 0
    copy_data = np.copy(data)
    for i in range(len(removed_pixels)):
        for j in range(np.size(copy_data,2)):
            copy_data[removed_pixels[i][0]][removed_pixels[i][1]][j] = 0

    return copy_data




def clipping(data,labels,wells,t_m,t_n):
    M = np.size(data,0)
    sample = []
    n_labels = []
    #separate the wells
    if wells == 2:
        sample.append(data[0:round(M/2)][:][:])
        sample.append(data[round(M/2)+1:][:][:])
        n_labels.append(labels)
        n_labels.append(labels)

    else: 
        sample.append(data)
        n_labels.append(labels)
    #downsize M and N dimension
    trimmed_samples = []
    for s in range(len(sample)):
        try:
            remove_rows= []
            remove_columns = []
            Tred = np.size(sample[s],2)
            mat = np.mean(sample[s],axis=2)
            Mred = np.size(mat,0)
            Nred = np.size(mat,1)
            threshold_m = np.mean(mat[:][:]) * Nred * t_m
            threshold_n = np.mean(mat[:][:]) * Mred * t_n
            for n in range(Nred):
                try:
                    if np.mean(mat[:][n]) <= threshold_n:
                        remove_columns.append(n)
                except:pass
            for m in range(Mred):
                try:
                    if np.mean(mat[m][:]) <= threshold_m:
                        remove_rows.append(m)
            # for n in range(Nred):
            #     try:
            #         if np.mean(mat[:][n]) <= threshold_n:
            #             remove_columns.append(n)
                except:pass 
            
            trim = np.delete(sample[s],remove_rows,0)
            trim = np.delete(trim,remove_columns,1)
            trimmed_samples.append(trim)
        except: pass
    return trimmed_samples,n_labels


def criterion_i(w_i,lamda,datar):

    mean = 1/lamda 
    upper_bound = (2*mean) + w_i 
    Mf = np.size(datar,0) 
    Nf = np.size(datar,1) 
    data = np.copy(np.mean(datar,axis=2))
    counter = 0
    for m in range(Mf):
        for n in range(Nf):
            if data[m,n] < 5:
                counter = counter + 1
            else: pass
    if counter > upper_bound:
        return 0
    else:
        return 1


def criterion_ii(w_ii,mu_m,mu_n,sigma_m,sigma_n,data):
    upper_bound_m = (mu_m + sigma_m*w_ii)
    upper_bound_n = (mu_n + sigma_n*w_ii)
    lower_bound_m = (mu_m - sigma_m*w_ii)
    lower_bound_n = (mu_n - sigma_n*w_ii)
    avdata = np.mean(data,axis=2)
    Mf = np.size(avdata,0) 
    Nf = np.size(avdata,1) 
    if ( lower_bound_m > Mf)  or (Mf > upper_bound_m) or (lower_bound_n > Nf) or (Nf > upper_bound_n):
        return 0
    else: 
        return 1

def criterion_iii(w_iiia,w_iiib,alpha,beta_b,beta_g,gamma,mu,data):
    total_DeltaV = []
    mean_G = mu + beta_g * gamma
    std_G = beta_g * np.pi / np.sqrt(6)
    upper_G = mean_G + std_G*w_iiib
    mean_B = alpha / (alpha + beta_b)
    var = alpha * beta_b / ((alpha + beta_b)**2 * (alpha + beta_b + 1))
    std_B = np.sqrt(var)
    upper_B = (mean_B + std_B)*2 + w_iiia
    sample = np.copy(data)
    M = np.size(data,0)
    N = np.size(data,1)
    T = np.size(data,2)
    DeltaV =  np.zeros((M, N, T))
    for t in range(T-1):
        for m in range(M):
            for n in range(N):
                DeltaV[m][n][t] = data[m][n][t+1] - data[m][n][t]
    total_DeltaV.append(DeltaV)

    def euclidean_distance(p, q):
        return math.sqrt(p*p + q*q)

    total_dV = np.mean(total_DeltaV[0],axis=2)
    total_V = np.mean(data,axis=2)
    set_V = []
    set_dV = []
    
    
    dV  = np.copy(total_dV)
    V  = np.copy(total_V)

    M = np.size(dV,0)
    N = np.size(dV,1)

    for m1 in range(M):
        for n1 in range(N):
            euclideanV, euclideandV = [],[]
            count = 0
            cumulativedV_euclidean = 0
            cumulativeV_euclidean = 0                   
            for m2 in range(M):
                for n2 in range(N):
                        cumulativedV_euclidean = cumulativedV_euclidean + abs(euclidean_distance(np.mean(dV[m1][n1]),np.mean(dV[m2][n2])))
                        cumulativeV_euclidean = cumulativeV_euclidean + abs(euclidean_distance(np.mean(V[m1][n1]),np.mean(V[m2][n2])))

                        count = count + 1

            euclideandV.append(cumulativedV_euclidean/count)
            euclideanV.append(cumulativeV_euclidean/count)
    set_V.append(np.mean(euclideanV))
    set_dV.append(np.mean(euclideandV))
    if (np.mean(set_V)>upper_G) or (np.mean(set_dV)>upper_B):
        return 0
    else: return 1


def pre_processing(data3D,labels,c,d,t,g,z_m,z_n,w_i,w_ii,w_iiia,w_iiib):
    Accepted_data = []
    Accepted_labels = []
    well_num = []
    inference_time_preproc = []
    for i in range(len(data3D)):
        try:
            print(" Preprocessing Progress:",i+1,"/",len(data3D),end='\r')      
            global M, N, T

            M = 78
            N = 56
            T = np.size(data3D[i],2)
            # Standard steps for all samples:
            mat = np.copy(data3D[i])
            # startI = time.time()
            well_num = well_detection(mat,0.95,5)
            # Pipeline controlled via RL module
            if c == True:
                mat = Concentration_Cleaning(mat,d,t)
            if g == True:
                mat,lab = clipping(mat,labels[i],well_num,z_m,z_n)
            if g == False:
                lab = labels[i]
            for j in range(len(mat)): 
                
                C1 = criterion_i(w_i,0.005,mat[j])
                C2 = criterion_ii(w_ii,28,30,15,15,mat[j])
                C3 = criterion_iii(w_iiia,w_iiib,2.5,30,260,np.euler_gamma,850,mat[j])
               
                if C1*C2*C3 == 1:
                    print("-> sample:",i+1,".",j," accepted")
                    Accepted_data.append(mat[j])
                    Accepted_labels.append(lab[j])
                # else: print("-> sample:",i+1,".",j,"rejected")
                else: pass
        except: pass
    return Accepted_data,Accepted_labels

def feature_extract(Accepted_data,Accepted_labels,tw_s,tw_d):
    window_size,window_step =  tw_s,tw_d
    startI = time.time()
    count = 0
    def sliding_window(data,time_window_size,time_window_step):
        Td = np.size(data,2)
        delta = []
        average_delta = []
        Td = 440
        for i in range(Td-1):
            delta.append(data[:,:,i+1] - data[:,:,i])
        for i in range(0,int(len(delta)-time_window_step),int(time_window_step)):
            average_delta.append(np.mean(delta[i:int(i+time_window_size)],axis = 0))
        return average_delta


    delta_data = []
    for i in range(len(Accepted_data)):
        delta_data.append(sliding_window(Accepted_data[i],window_size,window_step))

    pixel_array = []
    pixel_labels = []
    S = len(delta_data)
    
    for s in range(S):

        T = np.size(delta_data[s],0)
        N = np.size(delta_data[s],1)
        M = np.size(delta_data[s],2)
        # print(T,N,M)
        for n in range(N):
            for m in range(M):
                mini_pixel_array = []   
                for f in range(T):
                    mini_pixel_array.append(delta_data[s][f][n][m])
                pixel_array.append(mini_pixel_array)  
                pixel_labels.append(Accepted_labels[s])

    pairs = list(zip(pixel_array, pixel_labels))

    random.shuffle(pairs)
    X, Y = zip(*pairs)
    ratio = 0.2
    split_point = round(len(pixel_array)*ratio)

    test_X = X[0:split_point]
    test_Y = Y[0:split_point]
    train_X = X[split_point+1:] 
    train_Y = Y[split_point+1:] 

    count_0 = 0
    count_1 = 0
    count_2 = 0
    count_3 = 0
    for i in range(len(test_Y)):
        if test_Y[i] == 0:
            count_0 += 1
        if test_Y[i] == 1:
            count_1 += 1
    for i in range(len(train_Y)):
        if train_Y[i] == 0:
            count_2 += 1
        if train_Y[i] == 1:
            count_3 += 1
    # print(" ") 
    # print("___________________________________________")      
    # print(" ") 
    # print("Number of Negative samples in Train:",count_2)
    # print("Number of Positive samples in Train:",count_3)
    # print("Number of Negative samples in Test:",count_0)
    # print("Number of Positive samples in Test:",count_1)
    # print("___________________________________________")
    # print(" ") 
    # print("Fake Accuracy from Positives:",np.round((count_1/(count_1+count_0))*10000)/10000)
    # print("Fake Accuracy from Negatives:",np.round((count_0/(count_1+count_0))*10000)/10000)
    #print("Training  Testing shape:   ","(",str(len(train_X))+", "+str(len(train_X[0])),")",",","(",str(len(test_X))+" , "+str(len(test_X[0])),")")
    # print("Training labels shape:   ","(",str(len(train_Y))+", "+" 1",")")
    # print("Testing  pixels shape:   ","(",str(len(test_X))+" , "+str(len(test_X[0])),")")
    # print("Testing  labels shape:   ","(",str(len(test_Y))+" , "+" 1",")")
    size = len(train_X)+len(test_X)
    endI = time.time()
    inf = endI-startI
    return test_X, test_Y, train_X, train_Y
