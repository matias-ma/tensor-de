# the gen_GL_data class is copied from https://github.com/ivanpeng0414/nonparametric-estimation-via-variance-reduced-sketching
''''
Usage:
python generate.py N d --output foo.npy
where N is the number of data points and d is the dimension
returns: GL data of shape (N,d)
'''''
import numpy as np
import argparse
import torch

device = torch.device('cpu')

class gen_GL_data:
    def __init__(self, d, epsilon, left_domain, right_domain, Ttotal_1, tau, lamda, h_list,normal_type):
        self.d = d
        self.n_dims = d
        self.epsilon = epsilon
        self.left_domain = left_domain
        self.right_domain = right_domain
        self.Ttotal_1 = Ttotal_1
        self.tau = tau
        self.lamda = lamda 
        self.h_list = h_list
        self.normal_type = normal_type
        self.h = h_list[0]
        self.beta = 2/self.epsilon**2
        
        ### domain of the underlying density [left, right]
        ### generate density, always [-1,1]
    
    ### input corporate a h_list here, this could be different
    def grad(self, X):

        
        B = (X**3 - X)/self.lamda
        
        A = torch.zeros_like(X,device=device)
        A[:, 0, :] = (X[:,0,:])*self.lamda/self.h_list[0]**2 + (X[:,0,:] - X[:,1,:])*self.lamda/self.h_list[1]**2
        for i in range(1,self.d-1):
            A[:, i, :] = (X[:,i,:] - X[:,i-1,:])*self.lamda/self.h_list[i]**2 +\
                         (X[:,i,:] - X[:,i+1,:])*self.lamda/self.h_list[i+1]**2
                    
        A[:, -1, :] = (X[:,-1,:] - X[:,-2,:])*self.lamda/self.h_list[self.d-1]**2 +\
                      (X[:,-1,:])*self.lamda/self.h_list[self.d-1]**2

        DeltaU = A + B 
       
        return DeltaU
    
    def generate(self, N):

        Xinit = torch.zeros([N, self.d],device=device)
        # Xinit = torch.rand([N, self.d],device=device) * (self.right_domain - self.left_domain) + self.left_domain
        ## [left_domian, right_domain]

        
        tau_num_1 = int(self.Ttotal_1 / self.tau)

        num_repeat = 1 
        Xtmp = Xinit.unsqueeze(2)


        for _ in range(tau_num_1):
            noise = torch.randn([N, self.d, num_repeat],device=device)
            Xtmp = Xtmp - (self.grad(Xtmp)) * self.tau + self.epsilon * np.sqrt(self.tau) * noise

        X_output1 = Xtmp.mean(dim=2)
        
        
        X_train = X_output1
        X_train = X_train.cpu().detach().numpy()
        
        X_train_min = np.min(X_train)
        X_train_max = np.max(X_train)
        if X_train_min < self.left_domain or X_train_max > self.right_domain:
            raise Exception('Out of Boundary')

        print('success')
        X_shift = X_train/self.right_domain ## [-1,1]

        return X_shift ## always [-1,1]

# This part is new
def main():
    parser = argparse.ArgumentParser(description="Generate Ginzburg–Landau data")
    parser.add_argument('N', type=int, help='Number of data points')
    parser.add_argument('d', type=int, help='Dimension of each sample')
    # parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon noise scale')
    # parser.add_argument('--left', type=float, default=-1.0, help='Left domain bound')
    # parser.add_argument('--right', type=float, default=1.0, help='Right domain bound')
    # parser.add_argument('--Ttotal', type=float, default=1.0, help='Total simulation time')
    # parser.add_argument('--tau', type=float, default=0.01, help='Time step tau')
    # parser.add_argument('--lamda', type=float, default=1.0, help='Lambda interaction strength')
    # parser.add_argument('--h', type=float, default=1.0, help='Grid spacing h')
    # parser.add_argument('--normal_type', choices=['Uniform', 'Known'], default='Uniform', help='Normalization type')
    parser.add_argument('--output', type=str, default='data.npy', help='Output .npy filename')
    args = parser.parse_args()

    # Build h_list of length d
    h = 1/(args.d+1)
    h_list = [h] * args.d

    # using the same parameters as in the aforementioned repo
    gen = gen_GL_data(
        d=args.d,
        epsilon=4,
        left_domain=-2.5,
        right_domain=2.5,
        Ttotal_1=10,
        tau=1e-3,
        lamda=0.02,
        h_list=h_list,
        normal_type='Known'
    )

    data = gen.generate(args.N)
    np.save(args.output, data)
    print(f"Saved {args.N} samples of dimension {args.d} to '{args.output}'")

if __name__ == '__main__':
    main()
