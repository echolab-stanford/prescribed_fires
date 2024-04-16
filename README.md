# Prescribed and low-intensity/severity fires project

This repo has all the replication data for the low-intensity and severity project. In this project we estimate the effects of specific fire events on the future fire dynamics, this means both the probability of future fire, its intensity and severity, and the prevented emissions from fire treatments. To do this, we rely on two main approaches: _synthtetic control_ and _differences in differences_. This repo contains code to execute data processing and estimation of the described effects in the paper. 

## Library configuration and experiments
Balancing, estimation, and data management is done in Python. To easily run extraction from raw files (see more in [datasets](#datasets)) and run balancing and estimation, we use [Hydra][1], a configuration library that relies on a set of configuration files in YAML format to configure and re-run paper analysis. 

For instance, to build the main analysis dataset, we can use the following configuration file:

```yaml
root_path: "~/data"
save_path: "${root_path}/processed"
data_path: "${root_path}/raw"
template: "${root_path}/template.tif"

extract:
  frp:
    path: "${data_path}/modis/fire_archive_M-C61_403638.csv"
    save_path: "${save_path}/frp"
```

You can access to help to make your own configuration files using `python main.py --help`, or by re-using some of the configuration files in the `conf` folder. Remember Hydra will save the configuration files for every experiment in the `outputs` folder with the date and time of running. You can override this behavior, but is good for reproducibility. 

<details>
<summary>More on configuration files </summary>
</details>

## Running experiments

All experiments can run in a Docker-ized environemnt using the Pangeo [`pytorch-notebook`][2] image. To run the experiments, you can use the following command:

```bash
docker run -it --rm -v $(pwd):$PROJECT_HOME -p 8888:8888 pangeo/pytorch-notebook python main.py
```

In an HPC environment, a Singularity instance can also be deployed using the same Docker configuration, including the GPU configuration: 

```bash
#!/bin/bash
#SBATCH --job-name=test      
#SBATCH --gres=gpu:1       
#SBATCH --partition=gpu
#SBATCH -c 8
#SBATCH --time=0-01:00 

module load singularity

singularity exec --nv --cleanenv --bind /home/$USER:/run/user image.sif python main.py
```

>[!TIP]
>The code is relying on the use of GPU for the balancing estimation. You can change the `config/estimation.yaml` to use CPU instead, but this can change the computation time significantly. In our experiments, each balancing experiment took around 15 minutes in a NVIDIA V100 GPU 16 GB, which is Sherlock's smallest GPU. 

Alternatively, you can try to run the experiments in a local environment by first installing the required environment: 

```bash
mamba create -n prescribed python=3.10
mamba env update -n prescribed -f environment.yml
```

And then running the experiments in `main` with the defined configuration files in `conf/`. 


### Datasets

| Variable       | Source                                                  |   Time    |          Link           |
| -------------- | ------------------------------------------------------- | :-------: | :---------------------: |
| Wildfires      | Monitoring Trends in Burn Severity                      | 1987-2022 |        [MTBS][3]        |
| Fire severity  | ${\Delta}$NBR calculated from Landsat Collection 2[^1]       | 1987-2022 | [Planetary Computer][4] |
| Fire Intensity | MODIS Burned Area Products (FIRMS)                      | 2000-2022 |    [MODIS Firms][5]     |
| Weather        | PRISM fire climatology variables                        | 1985-2022 |       [PRISM][8]        |
| DEM            | Digital Elevation Model (slope is calculated by us)[^2] |     -     |       [Paper][6]        |
| Disturbances   | Disturbance Agents in California                        | 1987-2021 |     [Dataverse][7]      |
| Forest Cover   | NVDI calculation from Landsat Collection 2              | 1987-2022 | [Planetary Computer][4] |

[^1]: See our own library [`dnbr_extract`][9] that follows an offset calculation of the NBR index.
[^2]: The DEM variables are calculated using the `xarray-spatial` algorithms.

### Notes
See the amazing libraries that we rely on to do this work:
 - [CBPS](https://github.com/apoorvalal/covariate_balancing_propensity_scores)

<!-- References -->
[1]: https://hydra.cc/
[2]: https://github.com/pangeo-data/pangeo-docker-images?tab=readme-ov-file
[3]: https://www.mtbs.gov/direct-download
[4]: https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2#Example-Notebook
[5]: https://modis-fire.umd.edu/
[6]: https://www-nature-com.stanford.idm.oclc.org/articles/sdata201840
[7]: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CVTNLY
[8]: https://prism.oregonstate.edu/
[9]: https://github.com/echolab-stanford/dnbr_extract