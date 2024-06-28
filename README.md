![TB1](extraFiles/logo.jpeg)

## Table of Contents

- [Introduction](#Introduction)
- [Usage Instructions](#Usage-Instructions)
- [Contributing](#contributing)
- [Creators](#Creators)
- [Copyright and license](#Copyright-and-license)

### Introduction

This repository addresses the problem of safe and energy-efficient routing of last-mile electric freight vehicles. With the rising environmental footprint of the transportation sector and the growing
popularity of E-Commerce, freight companies are likely to benefit from optimal time-window-feasible tours that minimize energy usage while reducing traffic conflicts at intersections and thereby
improving safety. This problem is formulated as a Bi-criterion Steiner Traveling Salesman Problem with Time-Windows (BSTSPTW) with energy consumed and the number of left turns at intersections as the
two objectives while also considering regenerative braking capabilities. We first discuss an exact mixed-integer programming model with scalarization to enumerate points on the efficiency frontier for
small instances. For larger networks, an efficient local search-based heuristic is developed. This heuristic uses several operators to intensify and diversify the search process. The utility of the
proposed methods is demonstrated using a benchmark data set and real-world data from Amazon delivery routes in Austin, US. The results show that the proposed heuristics can generate near-optimal
solutions within reasonable time budgets, striking a balance between energy efficiency and safety under practical delivery constraints.

### Usage Instructions

- Before running the code, install the required packages using the following command:
  ```
  pip install -r requirements.txt
  ```
    - Requires Python>= 3.9.8.
    - Operating System: Ubuntu 20.04.3 LTS.
    - Requires CPLEX (Full Version) for running the MIP model.
- Download the sample instances from the
  following [(Link)](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/prateeka_iisc_ac_in/Ev82jXTC24FIn7YQRe18kBUBfpDAil-wexnAZuPRL0evvw?e=wTWnrt) and place them in the main
  directory.
- For running MIP model on benchmark instances, ensure the test instances are in folder "milpInputs". Run the following command:
  ```
  python mainIP.py
  ```
- For running local search on Amazon dataset, ensure the routes have been processed and saved in folder "./lsInputs/". A test instance (route Id 4946) has been provided. Run the following command:
  ```
  python mainLS.py
  ```
- For operator analysis, run the following command:
  ```
  python mainOperatorAnalysis.py
  ```
- For running a different instance, run
    ```
    python preprocessing.py
    ```

### Contributing

We welcome all suggestions from the community. If you wish to contribute or report any bug please contact the creaters or create an issue
on [issue tracking system](https://github.com/transnetlab/transit-routing/issues).

### Creators

- **Prateek Agarwal**
    - Ph.D. at Indian Institute of Science (IISc) Bengaluru, India.
    - Mail Id: prateeka@iisc.ac.in
    - <https://sites.google.com/view/prateek-agarwal/>

- **Debojjal Bagchi**
    - Ph.D. at The University of Texas at Austin.
    - Mail Id: debojjalb@utexas.edu
    - <https://debojjalb.github.io/>

### Copyright and license

The content of this repository is bounded by MIT License. For more information see
[COPYING file](https://github.com/transnetlab/transit-routing/blob/main/LICENSE)

### Acknowledgments

We would like to extend our gratitude to our co-authors, **Dr. Tarun Rambha** (tarunrambha@iisc.ac.in) and  **Dr. Venktesh Pandey** (vpandey@ncat.edu) for their invaluable support and guidance
throughout the project. We also thank **Dr. Vivek Vasudeva** (vivekvasudev@iisc.ac.in) for reviewing the code.

