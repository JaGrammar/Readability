
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import numpy as np


X_train = []
y_train = []
Query_id = []
array_train_x0 = []
array_train_x1 = []

def readDataset(path):
    print('Reading training data from file...')
    with open(path, 'r') as file:
        for line in file:
            split = line.split()
            y_train.append(int(split[0]))
            X_train.append(extractFeatures(split))
            Query_id.append(extractQueryData(split))
    print('Read %d lines from file...' %(len(X_train)))
    return (X_train, y_train, Query_id)  

def extractFeatures(split):
    '''
    Extract the query to document features used
    as input to the neural network
    '''
    features = []
    for i in range(2, 18):
        features.append(float(split[i].split(':')[1]))
    return features

def extractQueryData(split):
    '''
    Extract the query features from a dataset line
    Format:
    <query-id><document-id><inc><prob>
    '''
    queryFeatures = [split[1].split(':')[1]]
    return queryFeatures

def extractPairsOfRatedSites(y_train, Query_id):
    '''
    For each queryid, extract all pairs of documents
    with different relevance judgement and save them in
    a list with the most relevant in position 0
    '''
    pairs = []
    tmp_x0 = []
    tmp_x1 = []
    
    for i in range(0, len(Query_id)):
        for j in range(i+1, len(Query_id)):
            #Only look at queries with the same id
            if(Query_id[i][0] != Query_id[j][0]):
                break
            #Document pairs found with different rating
            if(Query_id[i][0] == Query_id[j][0] and y_train[i] != y_train[j]):
                #将最相关的放在前面,保持文档pair中第一个doc比第二个doc与query更相关
                if(y_train[i] > y_train[j]):
                    pairs.append([i, j])
                    tmp_x0.append(X_train[i])
                    tmp_x1.append(X_train[j])
                else:
                    pairs.append([j, i])
                    tmp_x0.append(X_train[j])
                    tmp_x1.append(X_train[i])
            if(Query_id[i][0] == Query_id[j][0] and y_train[i] == y_train[j]):
                print(i+1,j+1)
    #array_train_x0里和array_train_x1里对应的下标元素，保持前一个元素比后一个元素更相关
    array_train_x0 = np.array(tmp_x0)
    array_train_x1 = np.array(tmp_x1)
    
    print('Found %d document pairs' %(len(pairs)))
    
    return pairs,array_train_x0,array_train_x1


class Dataset(data.Dataset):
    
    def __init__(self,path):
        # 解析训练数据
        #readDataset(path)
        # pair组合
        self.datasize,self.array_train_x0,self.array_train_x1 = extractPairsOfRatedSites(y_train,Query_id)

    def __getitem__(self,index):
        data1 = torch.from_numpy(self.array_train_x0[index]).float()
        data2 = torch.from_numpy(self.array_train_x1[index]).float()
        #print(len(data1))
        return data1,data2
        
    def __len__(self):
        return self.datasize

#path = '/media/jun/ubuntu/RankNet/RankNet-master2/Data/Fold1/newtrain5.txt'

class RankNet(nn.Module):
    def __init__(self, inputs, hidden_size, outputs):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(inputs, hidden_size),
            #nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(0.2,  inplace=True),#inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
            nn.Linear(hidden_size, outputs),
            #nn.Sigmoid()
            )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_1, input_2):
        
        result_1 = self.model(input_1) #预测input_1得分
        result_2 = self.model(input_2) #预测input_2得分
        pred = self.sigmoid(result_1-result_2) #input_1比input_2更相关概率
        return pred
    
    def predict(self, input):
        result = self.model(input)
        return result

def train():
    #Training设置参数
    inputs = 16
    hidden_size = 10
    outputs = 1
    learning_rate = 0.15
    num_epochs = 100
    batch_size = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RankNet(inputs, hidden_size, outputs).to(device)
    #损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adadelta(model.parameters(), lr = learning_rate)
    
    datasize,array_train_x0,array_train_x1 = extractPairsOfRatedSites(y_train,Query_id)
    #print(array_train_x0)
    data1 = torch.from_numpy(array_train_x0).float()
    data2 = torch.from_numpy(array_train_x1).float()
    
    train_dataset = data.TensorDataset(data1,data2)
    #print(train_dataset)
    data_loader = data.DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = False,num_workers = 4)

    total_step = len(data_loader)
    print(total_step)
    
    for epoch in range(num_epochs):
        for i,(features, lables) in enumerate(data_loader):
            print('Epoch [{}/{}], Step [{}/{}]'.format(epoch+1, num_epochs, i+1, total_step))
            # Move tensors to configured device
            features = features.to(device)
            lables = lables.to(device)
            label_size = features.size()[0]
            pred = model(features,lables)
            loss = criterion(pred, torch.from_numpy(np.ones(shape=(label_size, 1))).float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    torch.save(model.state_dict(), '/media/jun/ubuntu/RankNet/RankNet-master2/Data/Fold5/model.ckpt')

def test():
    #test data
    test_path = '/media/jun/ubuntu/RankNet/RankNet-master2/Data/Fold5/test5.txt'
    # 超参
    inputs = 16
    hidden_size = 10
    outputs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RankNet(inputs, hidden_size, outputs).to(device)
    model.load_state_dict(torch.load('/media/jun/ubuntu/RankNet/RankNet-master2/Data/Fold5/model.ckpt'))
    print('Reading test data from file...')
    with open(test_path, 'r', encoding='utf-8') as f:
        features = []
        testdata_labels = []
        line_num = 0
        for line in f:
            line_num += 1
            toks = line.split()
            feature = []
            for label in toks[0]:
                testdata_labels.append(int(label))
            for tok in toks[2:]:
                _, value = tok.split(":")
                feature.append(float(value))
            features.append(feature)
        print('Read %d lines from file...' %(line_num))
        features = np.array(features)
     
    features = torch.from_numpy(features).float().to(device)
    predict_score = model.predict(features)
    results = predict_score.tolist()
    #print(results)
    #print(len(results[::2]))
    
    test_labels1=results[::2]
    test_labels2=results[1::2]

    pre_results=[]

    for i in range(0,500):
        if test_labels1[i]>test_labels2[i]:
            pre_results.append(1)
            pre_results.append(0)
        else:
            pre_results.append(0)
            pre_results.append(1)
    #print(len(pre_results))
    
    correct_num=[]
    for i in range(0,1000):
        if pre_results[i] == testdata_labels[i]:
            correct_num.append("Yes")

    correct_error = len(correct_num)/len(testdata_labels)
    print(correct_error)
            
                
    




if __name__ == '__main__':

    #Read training data
    X_train, y_train, Query_id = readDataset('/media/jun/ubuntu/RankNet/RankNet-master2/Data/Fold5/train5.txt')
    #Extract document pairs
    pairs = extractPairsOfRatedSites(y_train, Query_id)
    
    train()
    test()
    #Read testset
    #X_train, y_train, Query_id = readDataset('/media/jun/ubuntu/RankNet/RankNet-master2/Data/Fold1/newtest5.txt')
    #Extract document pairs
    #pairs = extractPairsOfRatedSites(y_train, Query_id)
    
    
    

