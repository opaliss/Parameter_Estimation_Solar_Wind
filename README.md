## Bayesian Inference and Global Sensitivity Analysis for Ambient Solar Wind Prediction

[Issan O, Riley P, Camporeale E, Kramer B (2023) *Bayesian Inference and Global Sensitivity Analysis for Ambient Solar Wind Prediction*. Arxiv. doi: arXiv:2305.08009.](https://arxiv.org/abs/2305.08009)

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
2. ACE Measurement data: available online at https://spdf.gsfc.nasa.gov/.
* Both data products are directly downloaded using *heliopy* package.

## Code Structure
**model_chain.py** - module to run the model chain *PFSS $\to$ WSA $\to$ HUX*. 
**

## Correspondence
Opal Issan (PhD student), University of California, San Diego. oissan@ucsd.edu

## License
MIT
 

