## How to add a point cloud

You need to have a point cloud with the three coordinates x, y, z and the verticality.

1. Open it in cloud compare and cut the point cloud to save only the admissible centers (the ones that will be selected afterwards). To do so, use to tool called `segment`.
2. Save the admissible centers as binary in data/centers.
3. Go to the [city_statistics](./city_statistics.ipynb) notebook, you will need to change to absolute path so that the jupyter kernel is in the top folder of the repository.
4. Change the `point_cloud_name` and run all cells. You can then add the new values obtained from the output's cells to the [init](./__init__.py) file. In total, the dictionaries `CENTERS`, `Z_GROUNDS` and `ROTATIONS` need to be updated.
5. You can check the visualizations cells of the notebook to see if everything is happening fine.
6. To use `RangeNet++` afterwards, you need to update the dictionnary `CITY_INFERANCE_FOLDER` in the [init](./__init__.py) file. 

You can now freely generate new samples from the point cloud that you have. For that, you need to use the command line `create_dataset`. If by any chance the pipeline gets broken, a notebook called [create_dataset](./create_dataset.ipynb) has been made to help debugging. 


## How to train RangeNet++

You can open the Jupyter Notebook directly in colab by clicking here:
<a href="https://colab.research.google.com/github/theovincent/3DPointCloudClassification/blob/rangenet/range_net/RangeNet%2B%2B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Consider restarting the runtime if a module is not found.

The step are explained in the notebook. You need to upload your data into the Colab session. The way it is done here is by using Google Drive.

For further details, please go and check the [documentation](https://github.com/PRBonn/lidar-bonnetal) of RangeNet++.


## How to infer and merge the predictions
The steps to follow are the following:
1. First generate the samples you would like to predict on by using the command line `create_dataset`.
2. Then generate the predictions with the weights that you have trained from the [previous section](#how-to-train-rangenet).
3. Merge the predictions by using the command line `merge_labels`.

It has been coded here:
<a href="https://colab.research.google.com/github/theovincent/3DPointCloudClassification/blob/rangenet/range_net/RangeNet%2B%2B_prediction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
Consider restarting the runtime if a module is not found.