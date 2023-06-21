

def clean_temp(data):   
    size_data = np.size(data,2)
    pixel_pattern = []
    for x in range(0,26):
        pixel_pattern.append(3*x + 1)
    for i in range(size_data):
        for a in range(1,55):
            for b in range(1,77):
                sum = data[b+1][a+1][i]+data[b+1][a+0][i]+data[b+0][a+1][i]+data[b-1][a+0][i]+data[b+0][a-1][i]+data[b-1][a-1][i]+data[b-1][a+1][i]+data[b+1][a-1][i]
                if (a in pixel_pattern) and (b in pixel_pattern):
                    data[b][a][i] = sum/8 
                else: pass
    return data


def clean_malfunctioning(data):
    M = np.size(data,0) 
    N = np.size(data,1) 
    T = np.size(data,2)
    for t in range(T):
        for n in range(N):
            for m in range(M):
                if data[m][n][t] > 990:
                    data[m][n][t] = 0
    return data

def well_detection(data,tuning_param,concecutive):
    global M,N,T 
    separated_data = []
    avdata = np.mean(data,axis=2)
    threshold = np.mean(avdata)*tuning_param
    crop_points = []
    crop_state = 0
    well_number = 1
    # STATE MACHINE to chek if 1 or 2 wells
    Z = 0
    for b in range(M):
        row_av =  np.mean(avdata[b][:])
        if crop_state == 0:
            if row_av > threshold:
                Z = Z+1
                if Z>=concecutive:
                    crop_state = 1                
            else: 
                Z = 0
                crop_state = 0
        if crop_state == 1:
            if row_av < threshold:
                Z = Z+1
                if Z >= concecutive:
                    crop_state = 2               
            else: 
                Z = 0
                crop_state = 1  
        if crop_state == 2:
            if row_av > threshold:
                Z = Z+1
                if Z >= concecutive:
                    well_number = 2
                    crop_state = 3                   
            else: 
                Z = 0
                crop_state = 2         
        else: pass
    # At this point if it is 2 wells, crop state will be 3
    return well_number

def Concentration_dimension(data,Y):
    Mat = np.mean(data,axis=2)
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
                    if Mat[i][j] < 100 :
                        pass
                    elif Mat[i][j] >= 100: 
                        Criterion1 = Criterion1 + 1
                else:
                    if Mat[i][j] < 100 :
                        pass
                    elif Mat[i][j] >= 100: 
                        Criterion2 = Criterion2 + 1
            elif Y == 1:
                if (i in RoIs[0]) and (j in RoIs[1]):
                    if Mat[i][j] < 100 :
                        pass
                    elif Mat[i][j] >= 100: 
                        Criterion1 = Criterion1 + 1
                else:
                    if Mat[i][j] < 100 :
                        pass
                    elif Mat[i][j] >= 100: 
                        Criterion2 = Criterion2 + 1           
    return Criterion1/Criterion2

def Activity_dimension(data):
    avdata = np.mean(data,axis=2)
    Mf = np.size(avdata,0) 
    Nf = np.size(avdata,1) 
    counter = 0
    for m in range(Mf):
        for n in range(Nf):
                counter = counter + avdata[m,n] 
    return counter/(76*56)


conpoints = []
actpoints = []
data = []
for i in range(len(data3D)):
    print("Progress:",i+1,"/",len(data3D),'\r')
    global M, N, T
    M = 78
    N = 56
    T = np.size(data3D[i],2)
    dataz = clean_temp(data3D[i])
    res_data = clean_malfunctioning(dataz)
    well_num = well_detection(res_data,0.95,5)
    con_ax = Concentration_dimension(res_data,well_num)
    act_ax = Activity_dimension(res_data)
    conpoints.append(con_ax)
    actpoints.append(act_ax)
    data.append(res_data)



