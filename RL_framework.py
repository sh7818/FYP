def RL_enviroment(train,group,data,labels,episodes,):
    repetitions = 30
    Ad = 0
    if train == False:
        data0, data1, data2, label0, label1, label2 = Grouping_kmeans(conpoints,actpoints,data,labels)
        G = [[data0,label0],[data1,label1],[data2,label2]]

        results = []

        # for i in range(3):
            # try:
        i = 1 # remove and replace with for loop to iteate all groups
        print("Testing Group:",i)
        print(" ")
        filename = f"Optim_path_Group_{i}.pickle"
        cR = 0
        ctp,ctn,cfp,cfn = 0,0,0,0
        with open(filename, "rb") as f:
            S = pickle.load(f)
            c,d,t,g,z_m,z_n,w_i,w_ii,w_iiia,w_iiib,tw_s,tw_d  = S
            Accepted_data,Accepted_labels = pre_processing(G[i][0],G[i][1],c,d,t,g,z_m,z_n,w_i,w_ii,w_iiia,w_iiib) 
            threshold = 0.5
            test_X, test_Y,train_X, train_Y= feature_extract(Accepted_data,Accepted_labels,tw_s,tw_d)
            acc, pred , vals = train_validate(test_X, test_Y, train_X, train_Y,0.5)
            TP, TN, FP, FN = calculate_confusion_matrix(pred , vals)
            results.append([i,acc,TP, TN, FP, FN])       
            # except: pass
                                
        return results
        


            

    else:    
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Q is equivalent to the the lookup values of the
        # corresponding accuracy for each given state.
        # 
        # The Agent is responsible for the states and actions.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initializations & Memory:
        Types = ["c", "d", "t","g", "z_m", "z_n","w_i", "w_ii", "w_iiia","w_iiib","tw_s", "tw_d"]     
        class States:
            def __init__(self):
                self.cases = {
                    Types[0]:  [True,False], Types[1]:   [3,13],      Types[2]:  [150,400],      # "c", "d", "t"  for smoothening function
                    Types[3]:  [True,False], Types[4]:   [0.001,0.1], Types[5]:  [0.001,0.1],  # "g", "z_m", "z_n" for clipping
                    Types[6]:  [-200,200],   Types[7]:   [1,5],      Types[8]:  [0.01,0.2],   # "w_i", "w_ii", "w_iiia" for criteria
                    Types[9]:  [-200,200],   Types[10]:  [1,7],       Types[11]: [1,7]        # "w_iiib","tw_s", "tw_d" feature extraction
                }            
            def get_case(self, argument):
                return self.cases.get(argument, [0,0])
        c,d,t = False,3,185                    # for smoothening function
        g,z_m,z_n = True,0.015,0.03                # for clipping
        w_i,w_ii,w_iiia,w_iiib = 0,1,0,0        # for criteria
        tw_s,tw_d = 3,3             # for feature extraction
        lock = 0                    # used in agend
        R_p = 0                  # initial reward
        gr0 = [c,d,t,g,z_m,z_n,w_i,w_ii,w_iiia,w_iiib,tw_s,tw_d]
        gr1 = [c,d,t,g,z_m,z_n,w_i,w_ii,w_iiia,w_iiib,tw_s,tw_d]
        gr2 = [c,d,t,g,z_m,z_n,w_i,w_ii,w_iiia,w_iiib,tw_s,tw_d]

        def Q(S,data,labels):
            repetitions = 1
            cR = 0
            c,d,t,g,z_m,z_n,w_i,w_ii,w_iiia,w_iiib,tw_s,tw_d = S
            Accepted_data,Accepted_labels = pre_processing(data,labels,c,d,t,g,z_m,z_n,w_i,w_ii,w_iiia,w_iiib)
            
            print("")
            print("Accepted samples:", len(Accepted_data))
            # for i in range(repetitions):    
            test_X, test_Y, train_X, train_Y, size = feature_extract(Accepted_data,Accepted_labels,tw_s,tw_d)
            if size<5000:
                cR = cR - 2  
            cR = cR + train_validate(test_X, test_Y, train_X, train_Y,0.5)

            R = cR/repetitions
            return R
        def Action_Value(S,data,Type,Type_ID):
            states = States()
            print("_____________________________________________________________________")
            print("Searching for: ",Type)
            print("RL State: ",S)
            Rewards = []
            # Actions:
            maxmin = states.get_case(Type)
            S_p = list(S)
            if isinstance(maxmin[0], bool):
                S_p[Type_ID] = False
                S_n = S_p
                Rewards.append([False,Q(S_n,data,labels)])
                S_p[Type_ID] = True
                S_n = S_p
                Rewards.append([True,Q(S_n,data,labels)])
                max_pair = max(Rewards, key=lambda pair: pair[1]) 
                print(">> Parameter: ", Type, "    , Optimal Value: ",max_pair[0]," , Accuracy: ", max_pair[1])
                return max_pair[0]
            else:
                upb = maxmin[1]
                lob = maxmin[0]
                if 0 < lob < 1:
                    x1 = int(round(lob*1000))
                    x2 = int(round(upb*1000))
                    for i in range(x1,x2,250):
                        try:
                            k = i/100
                            S_p[Type_ID] = k
                            S_n = S_p
                            Rewards.append([k,Q(S_n,data,labels)])
                        except: pass
                if 1 < abs(upb - lob) < 20 :
                    for k in range(lob,upb,2):
                        try:
                            S_p[Type_ID] = k
                            S_n = S_p
                            Rewards.append([k,Q(S_n,data,labels)])
                        except: pass
                if 50 < abs(upb - lob) < 101 :
                    for k in range(lob,upb,25):
                        try:
                            S_p[Type_ID] = k
                            S_n = S_p
                            Rewards.append([k,Q(S_n,data,labels)])
                        except: pass
                if 100 < abs(upb - lob) < 500 :
                    for k in range(lob,upb,100):
                        try:
                            S_p[Type_ID] = k
                            S_n = S_p
                            Rewards.append([k,Q(S_n,data,labels)])
                        except: pass
                        
                if 20 < abs(upb - lob) < 50 :
                    for k in range(lob,upb,10):
                        try:
                            S_p[Type_ID] = k
                            S_n = S_p
                            Rewards.append([k,Q(S_n,data,labels)])
                        except: pass
                else: 
                    for k in range(lob,upb):
                        try:
                            S_p[Type_ID] = k
                            S_n = S_p
                            Rewards.append([k,Q(S_n,data,labels)])
                        except: pass
                max_pair = max(Rewards, key=lambda pair: pair[1])   
                print(">> Parameter: ", Type, "    , Optimal Value: ",max_pair[0]," , Accuracy: ", max_pair[1])
                return max_pair[0]

        def Agent(data,Group,labels,S):
            if S == 0:
                if Group == 0: S = gr0;
                if Group == 1: S = gr1;
                if Group == 2: S = gr2;
            
            for i in range(len(Types)):
                # try:
                    S_param_optim = Action_Value(S,data,Types[i],i)
                    S[i] = S_param_optim
                # except: 
                    print(" ")
                    # print("Error for type:",Types[i])
                    # pass

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            print("Reinforcement learning pathways created")
            with open(f"Optim_path_Group_{Group}.pickle", "wb") as f:
                pickle.dump(S, f)
            return S
        

        # Here the Agent is controlled and run for the number of Episodes specified

        S_curr_ep = 0
        history_S = []
        for i in range(episodes):
           print("--->  Episode: ",i)
           S_next_ep =  Agent(data,group,labels,S_curr_ep)
           history_S.append(S_next_ep)
           S_curr_ep = S_next_ep