import matplotlib.pyplot as plt
from CNN_model import uniform_distribution
from SetParams import SplitList

"""Plot histogram of uniform distribution before and after split"""

if __name__ == '__main__':
    
    LF_train_size=2500
    HF_train_size=300
    u_range=[3,12]
    ti_range=[0.015, 0.25]
    yaw_range=[-30, 30]
    
    ### plot histogram ###
    
    u_list = uniform_distribution(size=LF_train_size, param_range=u_range)
    ti_list = uniform_distribution(size=LF_train_size, param_range=ti_range)
    yaw_list = uniform_distribution(size=LF_train_size, param_range=yaw_range)

    plt.hist(u_list)
    plt.show()
    plt.hist(ti_list)
    plt.show()
    plt.hist(yaw_list)
    plt.show()

    indices = SplitList(u_list, HF_train_size/LF_train_size)

    u_list = [u_list[index] for index in indices]
    ti_list = [ti_list[index] for index in indices]
    yaw_list = [yaw_list[index] for index in indices]

    plt.hist(u_list)
    plt.show()
    plt.hist(ti_list)
    plt.show()
    plt.hist(yaw_list)
    plt.show()
