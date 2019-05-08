import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import matplotlib.pyplot as plt

def encode_FEN(string):
    lsts = re.split('/', string)
    p = lsts[-1]
    del lsts[-1]
    lsts+= re.split(' ', p)
    b = np.zeros((4,8,8))
    if lsts[8] == 'b':
        for j in range(8):
            b_pointer = 0
            for i in range(len(lsts[j])):
                for num in range(2,9):
                    if lsts[j][i] == str(num):
                        b[0][j][b_pointer:b_pointer+num] = 0
                        b[1][j][b_pointer:b_pointer+num] = 0
                        b_pointer+= (num-1)
                
                if lsts[j][i] == 'p':
                    b[0][j][b_pointer]=1
                    b_pointer+=1
                
                if lsts[j][i] == 'r':
                    b[0][j][b_pointer] =5.25
                    b_pointer+=1
                
                if lsts[j][i]=='b':
                    b[0][j][b_pointer] = 3.5
                    b_pointer+=1
                
                if lsts[j][i]=='n':
                    b[0][j][b_pointer] =3.5
                    b_pointer+=1
                
                if lsts[j][i]=='q':
                    b[0][j][b_pointer] = 10
                    b_pointer+=1
                
                if lsts[j][i] =='k':
                    b[0][j][b_pointer] = 9
                    b_pointer+=1
                
                if lsts[j][i] == 'P':
                    b[1][j][b_pointer]=-1
                    b_pointer+=1
                
                if lsts[j][i] == 'R':
                    b[1][j][b_pointer] =-5.25
                    b_pointer+=1
                
                if lsts[j][i]=='B':
                    b[1][j][b_pointer] = -3.5
                    b_pointer+=1
                
                if lsts[j][i]=='N':
                    b[1][j][b_pointer] =-3.5
                    b_pointer+=1
                
                if lsts[j][i]=='Q':
                    b[1][j][b_pointer] = -10
                    b_pointer+=1
                
                if lsts[j][i] =='K':
                    b[1][j][b_pointer] = -9
                    b_pointer+=1
    #for white's turn I place the other thing on top
        if 'k' in lsts[9]:
            b[2][0][4] =2
    
        if 'q' in lsts[9]:
            b[2][0][3] = 2
            
        if 'K' in lsts[9]:
            b[3][7][4] = -2
            
        if 'Q'in lsts[9]:
            b[3][7][3] =-2    
    else:
        for j in range(7,-1,-1):
            b_pointer = 0
            for i in range(len(lsts[j])):
                for num in range(2,9):
                    if lsts[j][i] == str(num):
                        b[1][7-j][b_pointer:b_pointer+num] = 0
                        b[0][7-j][b_pointer:b_pointer+num] = 0
                        b_pointer+= (num-1)
                if lsts[j][i] == 'p':
                    b[1][7-j][b_pointer]=-1
                    b_pointer+=1
                
                if lsts[j][i] == 'r':
                    b[1][7-j][b_pointer] =-5.25
                    b_pointer+=1
                
                if lsts[j][i]=='b':
                    b[1][7-j][b_pointer] = -3.5
                    b_pointer+=1
                
                if lsts[j][i]=='n':
                    b[1][7-j][b_pointer] =-3.5
                    b_pointer+=1
                
                if lsts[j][i]=='q':
                    b[1][7-j][b_pointer] = -10
                    b_pointer+=1
                
                if lsts[j][i] =='k':
                    b[1][7-j][b_pointer] = -9
                    b_pointer+=1
                
                if lsts[j][i] == 'P':
                    b[0][7-j][b_pointer]=1
                    b_pointer+=1
                
                if lsts[j][i] == 'R':
                    b[0][7-j][b_pointer] =5.25
                    b_pointer+=1
                
                if lsts[j][i]=='B':
                    b[0][7-j][b_pointer] = 3.5
                    b_pointer+=1
                
                if lsts[j][i]=='N':
                    b[0][7-j][b_pointer] =3.5
                    b_pointer+=1
                
                if lsts[j][i]=='Q':
                    b[0][7-j][b_pointer] = 10
                    b_pointer+=1
                
                if lsts[j][i] =='K':
                    b[0][j][b_pointer] = 9
                    b_pointer+=1
        
        if 'k' in lsts[9]:
            b[3][7][4] =-2
        
        if 'q' in lsts[9]:
            b[3][7][3] = -2
            
        if 'K' in lsts[9]:
            b[2][0][4] = 2
            
        if 'Q'in lsts[9]:
            b[2][0][3] =2
    
    return torch.from_numpy(b)

class ChessConvNet(nn.Module):
    def __init__(self):
        """ Initialize the layers of your neural network
        """ 

        
        super(ChessConvNet, self).__init__()
        self.l1 = nn.Conv2d(in_channels=4,out_channels=10,kernel_size = 3)
        self.l2 = nn.MaxPool2d(kernel_size=2)
        self.l3 = nn.Conv2d(in_channels=10,out_channels=4,kernel_size = 3)
        self.l4 = nn.Linear(in_features=4, out_features=1)

    def set_parameters(self, kern1, bias1, kern2, bias2, kern3, bias3, fc_weight, fc_bias):
        """ Set the parameters of the network

        @param kern1: an (8, 2, 3, 3) torch tensor
        @param bias1: an (8,) torch tensor
        @param kern2: an (4, 8, 3, 3) torch tensor
        @param bias2: an (4,) torch tensor
        @param fc_weight: an (1, 4) torch tensor
        @param fc_bias: an (1,) torch tensor
        """
        
        self.l1.weight= nn.Parameter(kern1)
        self.l1.bias=nn.Parameter(bias1)
        #self.l2.weight = nn.Parameter(kern3)
        #self.l2.bias= nn.parameter(bias3)
        self.l3.weight= nn.Parameter(kern2)
        self.l3.bias = nn.Parameter(bias2)
        self.l4.weight = nn.Parameter(fc_weight)
        self.l4.bias = nn.Parameter(fc_bias)
        pass

    def intermediate(self, xb):
       
        n,a,b = xb.shape
        relu_on_convolve = F.relu(self.l1(xb))
        after_max_pool = self.l2(relu_on_convolve)
        convolve_2_relu = F.relu(self.l3(after_max_pool))
        
        return convolve_2_relu.view(1,4)

    def forward(self, xb):
        """ A forward pass of the neural network

        
        """
        n,a,b = xb.shape
        relu_on_convolve = F.relu(self.l1(xb.view(1,n,a,b)))
        #print(relu_on_convolve.shape)
        after_max_pool = F.relu(self.l2(relu_on_convolve))
        #print(after_max_pool.shape)
        convolve_2_relu = F.relu(self.l3(after_max_pool))
        #print(convolve_2_relu.shape)
        linear = (self.l4(convolve_2_relu.view(1,4)))
        
        return linear
    

def load_puzzle(pgn_handle):
    """
    Intended use case seen in fit():
    @param pgn_handle: file handle for your training file
    """
    board = chess.Board()
    game = chess.pgn.read_game(pgn_handle)
    if game is None:
        return None, None
    fen = game.headers['FEN']
    board.set_fen(fen)
    for j, mv in enumerate(game.mainline_moves()):
        if j == 0:
            board.push(mv)
        if j == 1:
            return board, mv

def fit(net,optimizer,n_epochs):
 
    epoch_loss = []
    boards = []
    moves = []
    with open('tactics.pgn') as pgn_handle:
            b, m = load_puzzle(pgn_handle)
            while b is not None:
                b, m = load_puzzle(pgn_handle)
                if b is None:
                    break
                boards.append(b)
                moves.append(m)

    for _ in range(n_epochs):
        print('epoch number', _+1)
        loss = torch.zeros(len(boards), dtype = torch.float)
        for i in range(len(boards)):
            
            score=torch.zeros(boards[i].legal_moves.count(), dtype = torch.float)
            y = torch.zeros(boards[i].legal_moves.count(), dtype= torch.float)
            boards[i].push(moves[i])
            fen1 = boards[i].fen()
            boards[i].pop()
            j = 0
            for move in boards[i].legal_moves:
                boards[i].push(move)
                fen2 = boards[i].fen()
                boards[i].pop()
                #print(encode_FEN(fen2).shape)
                score[j] = net(encode_FEN(fen2).float())
                if(fen1 == fen2):
                    y[j]=1
                j+=1
        
            #print(y)
            #print(score)
            loss[i] = sum((torch.exp(score)/sum(torch.exp(score)) -y)**2)
            #print(loss)
        
        epoch_loss.append(loss.sum())
   
        # gradients for the Tensors it will update (which are the learnable weights
        # of the model)
        optimizer.zero_grad()
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.sum().backward()
        
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
    
        
    torch.save(net.state_dict(), "model.pb")
    return epoch_loss
    #pass

def move(board):
    
    """
    @param board: a chess.Board
    @return mv: a chess.Move
    """
    i= 0
    score = torch.zeros(board.legal_moves.count(), dtype=torch.float)
    model = ChessConvNet()
    model.load_state_dict(torch.load('model.pb'))
    model.eval()
    moves = []
    
    for move in board.legal_moves:
        board.push(move)
        moves.append(move)
        score[i] = model(encode_FEN(board.fen()).float())
        board.pop()
        i+=1
        
    
    maxscore, index = torch.max(score,0)
 
    index = index.numpy()
  
    
    return moves[index]
    
    
   
