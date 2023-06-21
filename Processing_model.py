

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x

def train_validate(test_X, test_Y, train_X, train_Y,threshold):

    input_size = len(test_X[0])
    hidden_size = 128
    num_classes = 1

    num_epochs = 100

    X_train = torch.tensor(train_X, dtype=torch.float32)
    Y_train = torch.tensor(train_Y, dtype=torch.long)
    X_val = torch.tensor(test_X, dtype=torch.float32)
    Y_val = torch.tensor(test_Y, dtype=torch.long)

    model = DNN(input_size, hidden_size, num_classes)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    early_stoppage = 3
    tloss_h,vloss_h,acc_h = [],[],[]
    training_time = []
    inference_time = []
    patience = 0

    for epoch in range(num_epochs):
        if patience > early_stoppage: 
            # print("Early Stoppage applied on Epoch:",epoch)
            break
        else:
            startT = time.time()
            outputs = model(X_train)
            loss = criterion(outputs.squeeze().float(), Y_train.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            endT = time.time()
            tloss_h.append(loss)
            training_time.append(endT-startT)
            with torch.no_grad():
                startI = time.time()
                val_outputs = model(X_val)
                endI = time.time()
                inference_time.append(endI-startI)
                val_loss = criterion(val_outputs.squeeze().float(), Y_val.float())
                binary_pred = (val_outputs >= threshold).float()
                binary_pred = binary_pred.squeeze()
                Y_val = Y_val.float()
                accuracy = (binary_pred == Y_val).float().mean()
                try:
                    if val_loss.item()>(vloss_h[-1].item()+vloss_h[-2].item()+vloss_h[-3].item())/3:
                        patience +=1
                    else: patience = 0
                except: pass
                acc_h.append(accuracy)
                vloss_h.append(val_loss)
            # print("Epoch: {}, Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}".format(
            #     epoch, loss.item(), val_loss.item(), accuracy.item()))
    # print(" Train-Val complete. Max accuracy: ",max(acc_h))
    total_params = sum(p.numel() for p in model.parameters())
    total_size = total_params * 4  # Assuming 32-bit floating-point precision (4 bytes)  
    print(total_size)     
    return max(acc_h), binary_pred , Y_val


