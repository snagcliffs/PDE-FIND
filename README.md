The code and manuscript in this repository have a few errors worth noting that affect results:
1) Model selection uses the $\ell^2$ norm rather than the squared $\ell^2$ norm is shown in the manuscript.
2) TrainSTRidge passes the full data to STRidge rather than training partition of data as shown in the supplemental materials.
3) PolyDiff uses an even number of points.  This is not an error, per se, but was written unintentionally and is non-standard.

As this repository is not actively maintained, the code has been left as is.  An improved implementation may be found in the [PySINDy package](https://github.com/dynamicslab/pysindy). See notebook 10 in PySINDy/examples.  We also suggest the interested reader look into more recently developed weak forms of system and PDE identification as these lack the sensitivity to differentiation present in the current work.