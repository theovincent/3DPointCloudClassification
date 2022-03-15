## How to add a point cloud

You need to have a point cloud with the three coordinates x, y, z and the verticality.

1. Open it in cloud compare and cut the point cloud to save only the admissible centers (the ones that will be selected afterwards). To do so, use to tool called `segment`.
2. Save the admissible centers as binary in data/centers.
3. Go to the [city_statistics](./city_statistics.ipynb) notebook, you will need to change to absolute path so that the jupyter kernel is in the top folder of the repository.
4. Change the `point_cloud_name` and run all cells. You can then add the new values obtained from the output's cells to the [init](./__init__.py) file. In total, the dictionaries `CENTERS`, `Z_GROUNDS` and `ROTATIONS` need to be updated.
5. You can check the visualizations cells of the notebook to see if everything is happening fine.
6. To use `RangeNet++` afterwards, you need to update the dictionnary `CITY_INFERANCE_FOLDER` and the path `PATH_RANGE_NET` in the [init](./__init__.py) file. 

You can now freely generate new samples from the point cloud that you have. For that, you need to use the command line `create_dataset`. If by any chance the pipeline gets broken, a notebook called [create_dataset](./create_dataset.ipynb) has been made to help debugging. 