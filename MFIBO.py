from ModalAnalysis import ModalAnalysis as ma
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MFIBO:
    def __init__(self, model, w_exp, v_exp, n_vars, num_modes, max_iterations = 100, population_size = 100, ub=1, lb=0):
        self.lb = lb
        self.ub = ub
        self.population_size = population_size
        self.max_iter = max_iterations
        self.n_vars = n_vars
        self.model = model
        self.num_modes = num_modes
        self.w_exp = w_exp
        self.v_exp = v_exp
        self.log = np.zeros((max_iterations,))
        self.population = np.zeros((self.population_size, self.n_vars))
        self.best_solution = np.random.random((self.n_vars,))
    
    def calc_MF(self, v, element_number):
        pos1 = int(0.1+self.model.elements[element_number,0]-1)
        pos2 = int(0.1+self.model.elements[element_number,1]-1)
        dim = int(self.model.Ndfe/2)
        
        k = self.model.k[element_number]
        
        indices = list(range(dim*pos1,dim*(pos1+1)))
        indices.extend( list(range(dim*pos2,dim*(pos2+1))))

        phi_ij = v[indices, 1:self.num_modes]

        MFV_ij = np.abs(k@phi_ij);
        MF = np.sum(MFV_ij)
        return MF

    def cost_function(self, X):
        K=self.model.assembleStiffness(X)
        w, v = self.model.solve_eig(K, self.model.M)
        w=w[:self.num_modes]
        v=v[:,:self.num_modes]
        
        MAC=(np.sum((v*self.v_exp),axis=0)**2)/(np.sum(v*v,axis=0)*np.sum(self.v_exp*self.v_exp,axis=0))

        MDLAC=(np.abs(w-self.w_exp)/self.w_exp)**2

        cost = np.sum(1-MAC)+np.sum(MDLAC)
        return cost
    
    def plot_convergence(self):
        plt.plot(self.log,np.arange(int(self.max_iter)))
        plt.xlabel('Step')
        plt.ylabel('Cost Function Value')
        plt.title('Convergence Plot')
        plt.show()

    def run(self):
        self.best_fitness = self.cost_function(self.best_solution)
        MFa = np.zeros((self.n_vars,))

        for i in range(self.n_vars):
            MFa[i] = self.calc_MF(self.v_exp, i)

        for step in range(1,self.max_iter+1):
            print("Step {} : {}".format(step,self.best_fitness))
            self.log[step-1] = self.best_fitness
            
            K = self.model.assembleStiffness(self.best_solution)

            #solve the eigen value problem for the best solution
            _, phi = self.model.solve_eig(K, self.model.M)

            for i in range(self.population_size):
                for j in range(self.n_vars):
                    # MFc calculation
                    MFc = self.calc_MF(phi, j)
                    # MFR calculation
                    MFR=MFc / MFa[j]
                    rnd = np.random.random()
                    if rnd<=0.4:
                        self.population[i,j] = 1 - (1 - self.best_solution[j])*(MFR**np.random.random())
                    elif rnd>0.4 and rnd<=0.8:
                        if MFR>1:
                            self.population[i,j] = self.best_solution[j]-np.random.random()*(1-self.best_solution[j])/step
                        elif MFR<1:
                            self.population[i,j] = self.best_solution[j]+np.random.random()*self.best_solution[j]/step

                    else:
                        self.population[i,j] = self.best_solution[j]+(2*np.random.random()-1)*self.best_solution[j]
                    
                    if self.population[i,j]>self.ub:
                        self.population[i,j]=self.ub
                    elif self.population[i,j]<self.lb:
                        self.population[i,j]=self.lb
                
                #calculate the fitness and select the best solution for next iteration.
                cost = self.cost_function(self.population[i,:])
                
                if cost < self.best_fitness:
                    self.best_fitness = cost 
                    self.best_solution = self.population[i,:].copy()
            
        return self.log



if __name__=="__main__":
    file_name = '3D-data.xlsx'
    dimension = int(file_name[0])
    elements = pd.read_excel(file_name, sheet_name='Sheet1').values[:, 1:]
    nodes = pd.read_excel(file_name, sheet_name='Sheet2').values[:, 1:]
    aa = ma(elements, nodes, dimension)
    M=aa.assembleMass()

    x_exp=np.zeros(len(elements))
    x_exp[5]=0.35
    x_exp[23]=0.20
    x_exp[15]=0.4
    x_exp[10]=0.24

    K=aa.assembleStiffness(x_exp)
    w_exp, v_exp=aa.solve_eig(K,aa.M)

    num_modes=10

    w_exp=w_exp[:num_modes]
    v_exp=v_exp[:,:num_modes]

    optimizer = MFIBO(model=aa, w_exp = w_exp, v_exp = v_exp, n_vars=len(elements), population_size=40, max_iterations=100, ub=1, lb=0, num_modes=num_modes)
    optimizer.run()
    optimizer.plot_convergence()
    print(optimizer.best_solution)
