# foil noise

Prediction of noise levels on airfoil and inlet flow characteristics.

## About Dataset
NASA dataset obtained from a series of aerodynamic and acoustic tests of two and three-dimensional airfoil blade sections conducted in an anechoic wind tunnel. The data was obtained from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise

## Content
The NASA data set comprises different size NACA 0012 airfoils (n0012-il) at various wind tunnel speeds and angles of attack. The span of the airfoil and the observer position were the same in all of the experiments.

### Data creators:
Thomas F. Brooks, D. Stuart Pope and Michael A. Marcolini
NASA

### Attribute Information:

Input features:

- f: Frequency in Hertzs [Hz].
- alpha: Angle of attack (AoA, α), in degrees [°].
- c: Chord length, in meters [m].
- U_infinity: Free-stream velocity, in meters per second [m/s].
- delta: Suction side displacement thickness (𝛿), in meters [m].

Output:

- SSPL: Scaled sound pressure level, in decibels [dB].