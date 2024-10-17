## Bayesian Inference and Global Sensitivity Analysis for Ambient Solar Wind Prediction

[Issan O, Riley P, Camporeale E, Kramer B (2023). Bayesian Inference and Global Sensitivity Analysis for Ambient Solar Wind Prediction. Space Weather, 21, e2023SW003555. doi: 10.1029/2023SW003555.]([https://arxiv.org/abs/2305.08009](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023SW003555)

## Abstract 
The ambient solar wind plays a significant role in propagating interplanetary coronal mass ejections and is an important driver of space weather geomagnetic storms. A computationally efficient and widely used method to predict the ambient solar wind radial velocity near Earth involves coupling three models: Potential Field Source Surface, Wang-Sheeley-Arge (WSA), and Heliospheric Upwind eXtrapolation. However, the model chain has eleven uncertain parameters that are mainly non-physical due to empirical relations and simplified physics assumptions. We, therefore, propose a comprehensive uncertainty quantification (UQ) framework that is able to successfully quantify and reduce parametric uncertainties in the model chain. The UQ framework utilizes variance-based global sensitivity analysis followed by Bayesian inference via Markov chain Monte Carlo to learn the posterior densities of the most influential parameters. The sensitivity analysis results indicate that the five most influential parameters are all WSA parameters. Additionally, we show that the posterior densities of such influential parameters vary greatly from one Carrington rotation to the next. The influential parameters are trying to overcompensate for the missing physics in the model chain, highlighting the need to enhance the robustness of the model chain to the choice of WSA parameters. The ensemble predictions generated from the learned posterior densities significantly reduce the uncertainty in solar wind velocity predictions near Earth.

## Python Dependencies
1. Python >= 3.9.13
2. numpy >= 1.23.3
3. matplotlib >= 3.6.0
4. scipy >= 1.7.1
5. heliopy >= 0.15.3
6. sunpy >= 4.1.0
7. astropy >=5.1.1 
8. notebook >=6.4.3
9. pfsspy >= 1.1.2
10. pandas >= 1.3.2
11. h5netcdf >= 0.11.0
12. cdflib >= 0.3.20
13. streamtracer >= 1.2.0
14. multiprocess >= 0.70.14
15. requests >= 2.28.1
16. emcee >= 3.1.4

## Data
1. GONG CR Synoptic Maps: available online at https://gong.nso.edu/data/magmap/crmap.html.
2. ACE *in-situ* Measurements: available online at https://spdf.gsfc.nasa.gov/.
* *Note*: ACE *in-situ* messurements are directly imported in the code using the [*heliopy*](https://heliopy.readthedocs.io/en/0.15.3/) package and the GONG synoptic maps for CR2048-CR2058 are saved in the folder **GONG**.

## Code Structure
The code roadmap to run the UQ framework: sensitivity analysis $\to$ MCMC $\to$ ensemble forecasting: 

**SA_tools/generate_samples.py** $\to$ **SA_evaluate_samples** $\to$ **SA_analysis** $\to$ **MCMC_simulation** $\to$ **MCMC_analysis** $\to$ **ensemble_simulation**

### Main modules
1. **model_chain.py** - module to run the model chain PFSS $\to$ WSA $\to$ HUX.  
2. **HUX/hux_propagation.py** - module to run the HUX model. 
3. **ACE_tools/interpolate.py** - module to interpolate velocity field results at ACE's trajectory. 
4. **SA_tools/sobol.py** - module to compute Sobol' sensitivity indices via various MC estimators, e.g. Saltelli, Janon, and Jansen estimators.
5. **SA_tools/generate_samples.py** - module to sample the model chain parameter space via Latin Hypercube sampling (a quasi-MC method). 
6. **MCMC_simulation** - folder containing various modules (for different CRs) to run the MCMC [affine invariant ensemble sampler](https://emcee.readthedocs.io/en/stable/), i.e. emcee, to generate samples from the posterior distribution of the most influential parameters in the model chain. 

### Analysis notebooks
1. **model_chain_results** - folder containing Jupyter notebooks analyzing the model chain results for various CRs.
2. **SA_analysis** - folder containing Jupyter notebooks analyzing the first-order and total-order Sobol' sensitivity indices for various time periods. 
3. **MCMC_analysis** - folder containing Jupyter notebooks analyzing the MCMC posterior samples, MCMC convergence, and other heuristics.
4. **ensemble_simulation** - folder containing Jupyter notebooks to generate an ensemble forecast using MCMC posterior samples. 

## Correspondence
[Opal Issan](https://opaliss.github.io/opalissan/) (Ph.D. student), University of California San Diego. email: oissan@ucsd.edu

## License
MIT
 

