from ModalAnalysis import ModalAnalysis as ma
from MFIBO import MFIBO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    best_solutions=[]
    save_file="convergence_plots_2D_40_100_distributed"
    i=0
    while i<10:
        print(f'Run {i+1}/10')
        file_name = '2D-data.xlsx'
        dimension = int(file_name[0])
        elements = pd.read_excel(file_name, sheet_name='Sheet1').values[:, 1:]
        nodes = pd.read_excel(file_name, sheet_name='Sheet2').values[:, 1:]
        aa = ma(elements, nodes, dimension)

        x_exp=np.zeros(len(elements))
        x_exp[5]=0.35
        x_exp[23]=0.20
        x_exp[15]=0.40 #localized damage
        x_exp[10]=0.24

        K=aa.assembleStiffness(x_exp)
        w_exp, v_exp=aa.solve_eig(K,aa.M)

        num_modes=10

        w_exp=w_exp[:num_modes]
        v_exp=v_exp[:,:num_modes]

        optimizer = MFIBO(model=aa, w_exp = w_exp, v_exp = v_exp, n_vars=len(elements), population_size=40, max_iterations=100, ub=1, lb=0, num_modes=num_modes)
        log=optimizer.run()
        log[log<0.0]=1e-16
        # optimizer.plot_convergence()
        if optimizer.best_fitness<1e-5:
            best_solutions.append(optimizer.best_solution.copy())
            i+=1

            plt.yscale("log")
            plt.plot(np.array(log),'r-')
            plt.xlabel("iterations")
            plt.ylabel("cost")
            plt.title('MFIBO Convergence')
            # plt.show()
            plt.savefig(f'./{save_file}/convergence_{i+1}.png')
            plt.cla()
            print("*"*80)
    
    sols=np.stack(best_solutions)
    np.savetxt(f"./{save_file}/best_sols.csv",sols,delimiter=',')

if __name__=='__main__':
    main()