import multiprocessing as mp
import VideoProcessingMethods as vpm
import DataMethods as dm
from pathlib import Path
import concurrent.futures
import sys
from time import sleep

postInit_DF_path = Path(sys.arv[1])
# postInit_DF_path = Path('/global/home/groups/fc_xenopus/Lgaga_kve/20200720_Lgaga_604pm_cam2_1/Initialization_DF/20200720_Lgaga_604pm_cam2_1_PostInitializationDF.csv')

# params_df = dm.readCSV2pandasDF(postInit_DF_path)
#
# numChunks = len(params_df)
#
# print(params_df)
#
# for i in range(numChunks):
#     params_df = dm.readCSV2pandasDF(postInit_DF_path)
#     params_chunkRow = params_df.iloc[i]
#     print(params_chunkRow['ChunkName'])


# if __name__ == '__main__':
#     with mp.Pool(24) as p:
#         p.starmap(vpm.runFullVideoAnalysis, zip(range(numChunks), [postInit_DF_path]*numChunks))

# if __name__ == '__main__':
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         executor.map(vpm.runFullVideoAnalysis, zip(range(numChunks), [postInit_DF_path]*numChunks))

if __name__ == "__main__":
    arg = int(sys.argv[2])
    print(arg)
    sleep(arg)
    print(postInit_DF_path)
    vpm.runFullVideoAnalysis(arg, postInit_DF_path)


# parallel python scratch_15.py ::: $(seq 100)
