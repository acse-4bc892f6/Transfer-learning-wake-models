from CNN_model import Generator
import time
import numpy as np
import pandas as pd

"""
Generate 10 rounds of 100 simulations to calculate average time to generate 100 simulations.
Different random seed is used in each round.
"""

if __name__ == '__main__':

    # range of inflow conditions
    u_range=[3,12]
    ti_range=[0.015, 0.25]
    yaw_range=[-30, 30]

    # all PyWake wake models
    pywake_models = ['BGauss', 'CGauss', 'Jensen', 'Larsen', 'NGauss', 'SBGauss', 'TurboGauss', 'TurboJensen', 'ZGauss', 'Fuga']
    # generate 10 different random seeds
    seeds = np.linspace(10, 100, 10, dtype=int)
    # instantiate CNN
    gen = Generator(nr_input_var=3, nr_filter=16)
    # store times as 2d array
    timings = np.empty((len(seeds), len(pywake_models)))

    # generate simulations and measure time
    for i in range(len(seeds)):
        for j in range(len(pywake_models)):
            print(pywake_models[j], seeds[i])
            start = time.time()
            gen.create_pywwake_dataset(size=100, u_range=u_range, ti_range=ti_range, yaw_range=yaw_range,
                floris_name='FLORIS_input_jensen_1x3', model=pywake_models[j], seed=seeds[i])
            end = time.time()
            timings[i,j] = end-start

    # convert array to dataframe
    timings_df = pd.DataFrame(data=timings, index=pywake_models, columns=seeds)
    # add column to store average time
    timings_df['mean'] = timings_df.mean(axis=1)
    # save dataframe as csv
    timings_df.to_csv('timings100.csv')
